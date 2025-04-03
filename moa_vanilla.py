# adapt the codes from the original repo:  https://github.com/togethercomputer/MoA
# to run this file, also need to config the Together token key. ver: together-1.4.6

from utils_moa import *
from colorama import Fore,init
init(autoreset=True)
import json,datasets,argparse,joblib

parser = argparse.ArgumentParser()
parser.add_argument("--bench", type=str, default='gsm')
parser.add_argument("--samplecnt", type=int, default=1000)
parser.add_argument("--rounds", type=int, default=1) # this is the layers for moa, if set to 0, then moa is turned off
parser.add_argument("--fb", action="store_true") # whether to inject the opinions/feedbacks from proposers models
parser.add_argument("--shots", type=int, default=4)
parser.add_argument("--max_tokens", type=int, default=512)
args = parser.parse_args()
print("args===>", args)


# feel free to pick any models (open and closed both works here) here as proposers and aggregator

# small models version
reference_models = [
                    'mistralai/Mistral-7B-Instruct-v0.1', 
                    'mistralai/Mistral-7B-Instruct-v0.2', 
                    'mistralai/Mistral-7B-Instruct-v0.3',
                    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                    "google/gemma-2-9b-it"
                    ]

aggregator_model =  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# larger models can be tested also: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" # or use meta-llama/Llama-3.3-70B-Instruct-Turbo


if args.bench == 'gsm':
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

def calculate_reports(acc, f1, em, output, ii):
    pred = output.split('####')[-1].strip() 
    if args.bench == 'gsm':
        label = ii['answer'].split('####')[-1].strip()
        c = 1 if pred.replace(',','').replace('$','') == label else 0
        acc.append(c)
        print('metrics ==> accuracy based on {}===>'.format(len(acc)), round(sum(acc) / len(acc) * 100, 2), '\n')
        
    if args.bench == 'qa':
        em_ = exact_match_score(pred , ii["answer"])
        f1_, prec, recall = f1_score(pred, ii["answer"])
        em.append(float(em_))
        f1.append(f1_)
        print('metrics ==> em f1 based on {}===>'.format(len(em)), round(sum(em) / len(em) * 100, 2) , round(sum(f1) / len(f1) * 100, 2) )
        


em, f1,  = [], []
em_0, f1_0,  = [], []
em_fb, f1_fb,  = [], []

acc_raw, acc_moa_fb, acc_moa = [], [], []


for ii in ds[split].shuffle().select(range(args.samplecnt)):

    # prepare few-shots from train split
    messages = []
    if args.bench == 'gsm':
        messages.append({'role':'system', 'content':'give your answer to the question to solve grade school math problem. Think step by step , then print \'####\' and finally give your final answer.'})
        for i in ds['train'].shuffle().select(range(args.shots)):
            messages.append({'role': 'user',      'content': i["question"]})
            messages.append({'role': 'assistant', 'content': i["answer"]})
        messages.append({'role': 'user',      'content': ii["question"]})

    elif args.bench == 'he':
        messages.append({'role':'system', 'content':'Complete the given function in Python.'})
        for i in ds['train'].shuffle().select(range(args.shots)):
            messages.append({'role': 'user',      'content': i["question"]})
            messages.append({'role': 'assistant', 'content': i["answer"]})
        messages.append({'role': 'user',      'content': ii["question"]})

    elif args.bench == 'qa':
        messages.append({'role':'system', 'content':'give your answer to the question with regard to the references. Think step by step , then print \'####\' and finally give your final answer.'})
        for i in ds['train'].shuffle().select(range(args.shots)):
            ref = '\n'.join( [''.join(j) for j in i['context']['sentences'] ])
            messages.append({'role': 'user',  'content': i["question"] + '\nReferences:{}'.format(ref)  })
            messages.append({'role': 'assistant', 
                             'content': "The identified paragraphs start with {}. #### {}".format(' & '.join(i['supporting_facts']['title']),  i["answer"])})
        
        ref = '\n'.join( [''.join(j) for j in ii['context']['sentences'] ])
        messages.append({'role': 'user',      'content': ii["question"] + '\nReferences:{}'.format(ref)})
        
    # ready to request

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

    
    # the final layer

    # the baseline, raw, no references and no feedbacks from proposers
    output_0 = generate_with_references(
            model=aggregator_model,
            messages=messages,
            references=[],
            temperature=0,
            max_tokens=args.max_tokens,
            reference_models = []
            )

    # comparative method, have references but no feedbacks from proposers
    output_1 = generate_with_references(
            model=aggregator_model,
            messages=messages,
            references=references,
            temperature=0,
            max_tokens=args.max_tokens,
            reference_models = []
            )

    # comparative method, have references and feedbacks from proposers
    # output_fb = generate_with_references(
    #         model=aggregator_model,
    #         messages=messages,
    #         references=references,
    #         temperature=0,
    #         max_tokens=args.max_tokens,
    #         reference_models = reference_models 
    #         )


    try:
        print('='*20)
        print(Fore.WHITE + "oracle answer===>\n" + ii["answer"])
        print(Fore.YELLOW  + "output_0===>\n", output_0, '\n')
        print(Fore.MAGENTA + "output_1===>\n" + output_1, '\n')
        # print(Fore.GREEN +   "output_fb===>\n" + output_fb, '\n')
        print('='*20)
    except:
        pass

    
    

    # at each step during the iteration, we can monitor the comparative results.
    # but with more samples tested, the more accurate.

    print('raw:')
    calculate_reports(acc_raw, f1_0, em_0, output_0, ii)

    print('with references but no feedbacks:')
    calculate_reports(acc_moa, f1, em, output_1, ii)

    # print('with references and feedbacks:')
    # calculate_reports(acc_moa_fb, f1_fb, em_fb, output_fb, ii)


