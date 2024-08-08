from utils import *
from utils_qa_eval import *

from colorama import Fore,init
init(autoreset=True)
import json,datasets,argparse,joblib

parser = argparse.ArgumentParser()
parser.add_argument("--bench", type=str)
parser.add_argument("--samplecnt", type=int, default=1000)
parser.add_argument("--rounds", type=int)
parser.add_argument("--fb", action="store_true")
parser.add_argument("--shots", type=int, default=4)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=None)
# parser.add_argument("--sole_llm", type=str, default=None)
args = parser.parse_args()
print("args===>", args)



reference_models = ['microsoft/WizardLM-2-8x22B',
                    'mistralai/Mixtral-8x7B-Instruct-v0.1', 
                    'Qwen/Qwen2-72B-Instruct', 
                    'meta-llama/Meta-Llama-3-70B-Instruct-Turbo', 
                    'deepseek-ai/deepseek-llm-67b-chat']


# reference_models = [
#                     'mistralai/Mistral-7B-Instruct-v0.1', 
#                     'mistralai/Mistral-7B-Instruct-v0.2', 
#                     'mistralai/Mistral-7B-Instruct-v0.3',
#                     "meta-llama/Meta-Llama-3.1-8B-Instruct"]

aggregator_model =  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

if args.bench == 'beaver':
    ds = datasets.load_dataset('PKU-Alignment/BeaverTails-Evaluation')
    split = 'test'
elif args.bench == 'gsm':
    ds = datasets.load_dataset('openai/gsm8k', 'main')
    split = 'test'
elif args.bench == 'he':
    ds = datasets.load_dataset('openai/openai_humaneval')
    split = 'test'
    # from human_eval.data import write_jsonl, read_problems
    # problems = read_problems()
elif args.bench == 'qa':
    ds = datasets.load_dataset('hotpotqa/hotpot_qa',  'fullwiki') # ['distractor', 'fullwiki']
    split = 'validation'
elif args.bench == 'math':
    ds = datasets.load_dataset('lighteval/MATH',  'all') # ['distractor', 'fullwiki']
    split = 'test' # 5k

def cul(acc, f1, em, output, ii):
    pred = output.split('####')[-1].strip() 
    if args.bench == 'gsm':
        label = ii['answer'].split('####')[-1].strip()
        c = 1 if pred.replace(',','') == label else 0
        acc.append(c)
        print('metrics ==> accuracy based on {}===>'.format(len(acc)), round(sum(acc) / len(acc) * 100, 2), '\n')
        return c

    if args.bench == 'qa':
        em_ = exact_match_score(pred , ii["answer"])
        f1_, prec, recall = f1_score(pred, ii["answer"])
        em.append(float(em_))
        f1.append(f1_)
        print('metrics ==> em f1 based on {}===>'.format(len(em)), round(sum(em) / len(em) * 100, 2) , round(sum(f1) / len(f1) * 100, 2) )
        return None

fail_q = []
em, f1,  = [], []
em_0, f1_0,  = [], []
em_fb, f1_fb,  = [], []

acc0, acc_fb, acc = [], [], []
for ii in ds[split].shuffle(seed=args.seed).select(range(args.samplecnt)):
    messages = []
    if args.bench == 'gsm':
        messages.append({'role':'system', 'content':'give your answer to the question to solve grade school math problem. Think step by step , then print \'####\' and finally give your final answer.'})
        # messages.append({'role':'user', 'content':'here are {} examples for your information.'.format(args.shots)})
        #messages.append({'role':'assistant', 'content': 'ok'})
        for i in ds['train'].shuffle(seed=3333).select(range(args.shots)):
            messages.append({'role': 'user',      'content': i["question"]})
            messages.append({'role': 'assistant', 'content': i["answer"]})
        # messages.append({'role':'user', 'content':'examples end, now begin your task to answer this question:'})
        #essages.append({'role':'assistant', 'content': 'ok'})
        messages.append({'role': 'user',      'content': ii["question"]})

    elif args.bench == 'he':
        messages.append({'role':'system', 'content':'Complete the given function in Python.'})
        # messages.append({'role':'user', 'content':'here are {} examples for your information.'.format(args.shots)})
        #messages.append({'role':'assistant', 'content': 'ok'})
        for i in ds['train'].shuffle(seed=3333).select(range(args.shots)):
            messages.append({'role': 'user',      'content': i["question"]})
            messages.append({'role': 'assistant', 'content': i["answer"]})
        # messages.append({'role':'user', 'content':'examples end, now begin your task to answer this question:'})
        messages.append({'role': 'user',      'content': ii["question"]})

    # if args.bench == 'beaver':
    #     messages.append({"role": "user", "content": ii["prompt"]})
    #     messages.append({"role": "user", "content": completion_user_prompt})

    elif args.bench == 'qa':
        messages.append({'role':'system', 'content':'give your answer to the question with regard to the references. Think step by step , then print \'####\' and finally give your final answer.'})
        for i in ds['train'].shuffle(seed=3333).select(range(args.shots)):
            ref = '\n'.join( [''.join(j) for j in i['context']['sentences'] ])
            messages.append({'role': 'user',  'content': i["question"] + '\nReferences:{}'.format(ref)  })
            messages.append({'role': 'assistant', 
                             'content': "The identified paragraphs start with {}. #### {}".format(' & '.join(i['supporting_facts']['title']),  i["answer"])})
        
        ref = '\n'.join( [''.join(j) for j in ii['context']['sentences'] ])
        messages.append({'role': 'user',      'content': ii["question"] + '\nReferences:{}'.format(ref)})
        

    prev_references = []

    if args.rounds > 0:
        for i_round in range(args.rounds):
            print(Fore.CYAN + '\nrounds:{}'.format(i_round))
            references = []

            for reference_model in reference_models:
                print(Fore.YELLOW + reference_model+'====>')
                reference = generate_with_references(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature= 0 ,
                    max_tokens=args.max_tokens,
                    reference_models= reference_models if args.fb else []
                )

                if reference is not None:
                    references.append(reference)
                    try:
                        print(Fore.BLUE + reference) # .encode('utf-8', errors='ignore')
                    except:
                        pass

                else:
                    print(Fore.RED + reference_model)
                
            if i_round < args.rounds - 1:
                prev_references = references
                references = []
    else:
        references = []


    output_0 = generate_with_references(
            model=aggregator_model,
            messages=messages,
            references=[],
            temperature=0,
            max_tokens=args.max_tokens,
            reference_models = []
            )


    output_1 = generate_with_references(
            model=aggregator_model,
            messages=messages,
            references=references,
            temperature=0,
            max_tokens=args.max_tokens,
            reference_models = []
            )

    # output_fb = ""
    output_fb = generate_with_references(
            model=aggregator_model,
            messages=messages,
            references=references,
            temperature=0,
            max_tokens=args.max_tokens,
            reference_models = reference_models 
            )


    try:
        print(Fore.WHITE +   "output_0===>\n", output_0)
        print(Fore.MAGENTA + "output_1===>\n" + output_1)
        print(Fore.GREEN +   "output_fb===>\n" + output_fb)
    except:
        pass

    print(Fore.YELLOW + "oracle answer===>\n" + ii["answer"])
    

    print('without moa')
    c = cul(acc0, f1_0, em_0, output_0, ii)


    print('with moa')
    c = cul(acc, f1, em, output_1, ii)

    print('with moa & fb')
    c = cul(acc_fb, f1_fb, em_fb, output_fb, ii)


'''
7b level:

without moa
metrics ==> accuracy based on 129===> 93.8

with moa
metrics ==> accuracy based on 129===> 79.84

with moa & fb
metrics ==> accuracy based on 129===> 82.17
'''