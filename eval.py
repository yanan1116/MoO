import datasets,argparse,torch,glob,os,sys,json, random, time
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']='1'
from transformers import AutoTokenizer
from colorama import Fore,init
from tqdm.rich import tqdm
init(autoreset=True)
from utils_moo import *

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default='')
parser.add_argument("--sft_path", type=str, default='')
parser.add_argument("--llm_name", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
# parser.add_argument("--shots", type=int, default=4)
parser.add_argument("--testsize", type=int, default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--openai", action="store_true")
parser.add_argument("--specsft", type=int, default=None)
parser.add_argument("--split", type=str, default='test')
parser.add_argument("--save_pred", action="store_true")
parser.add_argument("--maxlen", type=int, default=None)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
parser.add_argument("--sft_shuffle", action="store_true")
parser.add_argument("--quant", action="store_true")
args = parser.parse_args()
print(args)


ds = datasets.load_dataset(args.ds)
print("ds===>", ds)



def compute_score(responses, labels, sft_epoch):
    if 'aqua' in args.ds.lower() or 'gsm8k' in args.ds.lower() or 'mmlupro' in args.ds.lower():
        acc = []
        for response, answer in zip(responses, labels):
            if 'gpt' in args.llm_name:
                try:
                    output = response.choices[0].message.content.strip()
                except:
                    output = ""

            else:
                output = response.outputs[0].text
            pred = output.split('####')[-1].strip() 
            label = answer.split('####')[-1].strip()
            c = 1 if pred.replace(',','') == label else 0
            acc.append(c)

            if args.debug:
                try:
                    print(Fore.GREEN + answer )
                    print(Fore.YELLOW + output )
                    print('check:', c)
                    print('-'*30 + '\n')
                except:
                    pass
        print("resultttttt ===>",args, sft_epoch, round(sum(acc) / len(acc), 4))

    elif "hotpotqa/hotpot_qa" == args.ds:
        em, f1,  = [], []
        for response, answer in zip(responses, labels):
            if 'gpt' in args.llm_name:
                try:
                    output = response.choices[0].message.content.strip()
                except:
                    output = ""
            else:
                output = response.outputs[0].text
            pred = output.split('####')[-1].strip() 
            em_ = exact_match_score(pred , answer.split('####')[-1].strip() )
            f1_, prec, recall = f1_score(pred, answer.split('####')[-1].strip() )
            em.append(float(em_))
            f1.append(f1_)

            if args.debug:
                try:
                    print(Fore.GREEN + answer + '\n' + Fore.YELLOW + output + "\n" + '-'*30 + '\n')
                except:
                    pass
        print("resultttttt ===>",args, sft_epoch, 
            'metrics ==> em f1 based on {}===>'.format(len(em)), round(sum(em) / len(em) * 100, 2) , round(sum(f1) / len(f1) * 100, 2) )

    elif 'math' in args.ds.lower():
        acc = []
        for response, solution in zip(responses, labels):
            if 'gpt' in args.llm_name:
                try:
                    output = response.choices[0].message.content.strip()
                except:
                    output = ""
            else:
                output = response.outputs[0].text            
            # pred = parse_answer(response)
            solution_pure = remove_boxed(last_boxed_only_string(solution))

            label = extract_math_answer(solution, False)        
            pred = extract_math_answer(output, False)

            # print(solution_pure, '<====>', label, '=====>', pred)
            # print(Fore.YELLOW + pred)
            # print(Fore.GREEN + label)
            # print('*'*30 + '\n')
            c = 1 if is_equiv(pred, label) else 0
            acc.append(c)
            if args.debug:
                try:
                    print(Fore.YELLOW + output)
                    print(Fore.BLUE + pred)
                    
                    print(Fore.MAGENTA + solution)
                    print(Fore.GREEN + solution_pure)
                    print('-'*30 + '\n')
                except:
                    pass
        print("resultttttt ===>",args, sft_epoch, round(sum(acc) / len(acc), 4))



prompts, labels, questions = [], [], []

if args.testsize:
    ds_for_pred = ds[args.split].select(range(args.testsize))
else:
    ds_for_pred = ds[args.split]

for i in ds_for_pred:
    assert 'SOLUTION:' in i['text'] and 'QUESTION:' in i['text']

    # few-shots / zero-shots + moa (pure prompting)
    if 'now start write your solution below.' in i['text']:
        split_s = 'now start write your solution below.\nSOLUTION:'
        assert split_s in i['text']
        prompts.append(i['text'].split(split_s)[0] + split_s )
        labels.append(i['text'].split(split_s)[-1])
        questions.append('????')     

    # moa
    elif 'OPINIONS END' in i['text'] and 'OPINIONS START' in i['text']:
        split_s = 'OPINIONS END\nSOLUTION:'
        assert split_s in i['text']
        prompts.append(i['text'].split(split_s)[0] + split_s )
        labels.append(i['text'].split(split_s)[-1])
        questions.append(i['text'].split('OPINIONS START')[0])

    # few-shots (pure prompting)
    elif 'Examples starts>>>' in i['text'] and '<<<Examples ends' in i['text']:
        prompts.append(i['text'].split('<<<Examples ends')[0] + '<<<Examples ends' \
            + i['text'].split('<<<Examples ends')[-1].split('SOLUTION:')[0] + 'SOLUTION:')
        labels.append(i['text'].split('<<<Examples ends')[-1].split('SOLUTION:')[-1])
        questions.append(i['text'].split('<<<Examples ends')[-1].split('SOLUTION:')[0].split('QUESTION:')[-1])

    elif i['text'].count('QUESTION:') == 1 and i['text'].count('SOLUTION:') == 1:
        split_s = 'SOLUTION:'
        prompts.append(i['text'].split(split_s)[0] + split_s)
        labels.append(i['text'].split(split_s)[-1])
        questions.append(i['text'].split(split_s)[0])
    else:
        raise ValueError('text error===>{}'.format(i['text']))


print("ds:", len(prompts))
assert len(labels) == len(prompts) and len(labels) == len(questions)
if args.debug:
    print('labels>>>>>>>>>>>>>>>')
    for i in labels:
        print(i)
        print('-'*30)

    print('prompts>>>>>>>>>>>>>>>')
    for i in prompts:
        print(i)
        print('-'*30)        

if 'gpt' in args.llm_name:

    from openai import AzureOpenAI
    client_azure = AzureOpenAI(
      azure_endpoint = "https://responsible-ai.openai.azure.com/", 
      api_version= "2024-10-01-preview",
    )
    responses = []
    for prompt in tqdm(prompts):
        response = client_azure.chat.completions.create(
          model=args.llm_name, # gpt-4o-mini gpt-35-turbo
          #response_format={ "type": "json_object" },
          messages=[
            {"role": "system", "content": "You are a helpful assistant designed solve arithmetic and mathematics problems."},
            {"role": "user", "content": prompt}
          ]
        )
        responses.append(response)
    compute_score(responses, labels, 'nosft')
else:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model= args.llm_name, 
                dtype='float16', 
                max_model_len = args.maxlen if args.maxlen else None, 
                tensor_parallel_size= torch.cuda.device_count(),
                #pipeline_parallel_size = torch.cuda.device_count(),
                gpu_memory_utilization= args.gpu_memory_utilization, 
                #seed=None,
                trust_remote_code=True,
                quantization= "bitsandbytes" if args.quant or 'bnb-4bit' in args.llm_name else None, 
                load_format= "bitsandbytes" if args.quant or 'bnb-4bit' in args.llm_name else "auto", 
                enforce_eager=True, 
                enable_lora=True if args.sft_path else False,
                tokenizer_mode= "mistral" if args.llm_name.startswith('mistralai') else 'auto',
                #cpu_offload_gb = 0 if args.quant or 'bnb-4bit' in args.llm_name else 16, 
                #swap_space=16
            )   
         

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        n=1, 
        stop=["USER:", "ASSISTANT:", "QUESTION:", "SOLUTION:", "###Instruction", "Response:", "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", ">>>", "<<<"], 
        #logprobs=20
        #repetition_penalty=1.7
    )

    if args.sft_path:
        sft_llms_path = glob.glob("{}/checkpoint-*".format(args.sft_path))
        assert sft_llms_path

        if args.sft_shuffle:
            random.shuffle(sft_llms_path)
        else:
            sft_llms_path.sort(key=os.path.getmtime)

        print("sft_llms===>\n", sft_llms_path)
        for ft in sft_llms_path:
            sft_epoch = int(ft.split('-')[-1])
            if args.specsft and sft_epoch != args.specsft:
                continue
            print('\n\nevaluation on {}'.format(ft))

            responses = llm.generate(prompts,
                                    sampling_params,
                                    lora_request=LoRARequest("llama31_adapter_{}".format(sft_epoch), sft_epoch, ft)
                                    )
            compute_score(responses, labels, sft_epoch)


    else:
        t1 = time.time()
        responses = llm.generate(prompts,
                                sampling_params
                                )   
        t2 = time.time() 
        print("time elapse:", int(t2-t1) )
        compute_score(responses, labels, 'nosft')
        
        if args.save_pred:
            result = [{'prompt_ori':p_ori, 'prompt':p, "label": l, "response": r.outputs[0].text.strip()} for p_ori, p, l, r in zip(questions, prompts, labels, responses)]
            with open("/home/ubuntu/moa/moa_csvs/df__{}__{}__{}.json".format(args.ds.split('/')[-1].lower(), args.llm_name.split('/')[-1], args.split), "w") as f:
                json.dump(result, f, indent=4)

