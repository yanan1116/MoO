import datasets,argparse,torch,json,sys,random,glob,os

def org_ds_dic(jss):
    assert len(set([len(js) for js in jss])) == 1

    ds_dic = {'text':[]}

    for i in range(len(jss[0])):
        enums = [js[i] for js in jss]
        assert len(set([i['prompt_ori'] for i in enums]))==1
        assert len(set([i['label'] for i in enums]))==1

        prompt_ori = list(set([i['prompt_ori'] for i in enums]))[0]
        label = list(set([i['label'] for i in enums]))[0]
        bl = [i['response'] for i in enums]
        opinions = '\n'.join(['>>>{}: #### {}'.format(ix, b) for ix, b in enumerate(bl)])
        text = "QUESTION:{}\nOPINIONS START\n{}\nOPINIONS END\nSOLUTION:{}".format(prompt_ori, opinions, label)
        ds_dic['text'].append(text)

    return datasets.Dataset.from_dict(ds_dic)


############## moa dataset ########################

# mmlu
benchmark = 'mmlupro'
for split in ['test', 'train']: 
    jss = []
    for llm in ['Phi-3.5-mini-instruct-bnb-4bit',
                 'Llama-3.1-8B-Instruct', 'Meta-Llama-3-8B-Instruct', 'mistral-7b-instruct-v0.3-bnb-4bit',
                'mistral-7b-instruct-v0.2-bnb-4bit', 'Mistral-Nemo-Base-2407-bnb-4bit',
                'Llama-3.2-3B-Instruct',  'Llama-3.2-1B-Instruct', 
                'gemma-2-9b-it-bnb-4bit']:
        jss.append(json.load(open("./moa_csvs/df__{}_shots_8__{}__{}.json".format(benchmark, llm, split), 'r', encoding='utf-8')))


# aqua
benchmark = 'math'
for split in ['test', 'train']: 
    files = glob.glob("./moa_csvs/df__{}__*__{}.json".format(benchmark, split))
    jss = []
    for file in files:
        if  'Llama-3.1-8B-Instruct' in file or 'Llama-3.2-3B-Instruct' in file or 'llama-3-8b-Instruct-bnb-4bit' in file:
            jss.append(json.load(open(file, 'r', encoding='utf-8')))

    print("jss cnt:", len(jss))

    ds_push = org_ds_dic(jss)
    ds_push.push_to_hub('yananchen/{}_moa_llama_31_32_3'.format(benchmark), 
                                        split = split, 
                                        token='')





################################



benchmark = 'aqua'
ds = datasets.load_dataset("yananchen/{}_moa".format(benchmark))
for split in ['test', 'train']:
    ds_dic = {'text':[]}
    for ii in ds[split]:
        
        body = ii['text'].split('OPINIONS START\n')[-1].split('OPINIONS END\n')[0]

        bl = []
        for i in body.split('>>>'):
            if not i:
                continue
            bl.append(i.split('####')[-1].strip())
        assert len(bl) == 7

        body_rm_cot = '\n'.join(['>>>{}: #### {}'.format(ix, b) for ix, b in enumerate(bl)])

        text_rm_cot = ii['text'].split('OPINIONS START\n')[0] + 'OPINIONS START\n' + body_rm_cot + '\nOPINIONS END\n' + ii['text'].split('OPINIONS END\n')[-1] 
        ds_dic['text'].append(text_rm_cot)
    datasets.Dataset.from_dict(ds_dic).push_to_hub('yananchen/{}_moa_rm_cot'.format(benchmark), split = split, 
                                token='')











######################################### construct few shots ######################################### 
import datasets,argparse,torch,json,sys,random,glob,os

def process_hotpotqa(example):
    ref = '\n'.join( [''.join(j) for j in example['context']['sentences'] ])
    prompt = example["question"] + '\nReferences:{}'.format(ref) 
    completion = "The identified paragraphs start with {}. #### {}".format(' & '.join(example['supporting_facts']['title']),  example["answer"])
    return {'question':prompt, 'answer': completion}

def process_aqua_rat(example):
    prompt = example['question'] + '\noptions: ' + ' '.join(example['options'])
    completion = example['rationale'] + " #### " + example['correct']
    return {'question':prompt, 'answer': completion}


def process_mmlu_pro(example):
    prompt = example['question']  + '\noptions:\n'
    choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    #choices_dic = {ix:c for ix, c in enumerate(choices)}
    for i, opt in enumerate(example['options']):
        prompt += "{}. {}\n".format(choices[i], opt)
    completion = example['cot_content'].replace('A:','') + " #### " + example['answer']
    return {'question':prompt, 'answer': completion}


def pick_shots(example, shots, ds_name, text_in, text_out):
    template = 'QUESTION:{}\nSOLUTION:{}\n'

    if shots >= 0:
        if ds_name in ['lighteval/MATH', 'lighteval/MATH-Hard']:
            sys = "Given a mathematics or algebraic  problem, Think step by step and determine the answer. Simplify your answer as much as possible."
        elif ds_name in ['TIGER-Lab/MMLU-Pro']:
            sys = "Given a multiple choice question, Think step by step , then print \'####\' and finally give your final answer."
        else:
            sys = "Given a mathematics or algebraic problem, Think step by step , then print \'####\' and finally give your final answer."

        if shots > 0:
            sys += '\n\nExamples starts>>>\n'
            # if ds_name == 'TIGER-Lab/MMLU-Pro':
            #     ds_sel = ds['train'].shuffle().select(range(shots)) # .filter(lambda x: x['category']==example['category'])
            # else:
            ds_sel = ds['train'].shuffle().select(range(shots)) # .filter(lambda x: x[text_in] != example[text_in])
            for i in ds_sel:
                sys += template.format(i[text_in], i[text_out])
            sys += '<<<Examples ends\n\n'

    else:
        sys = ""

    sys += template.format(example[text_in], example[text_out])
    example['text'] = sys
    return example


######################
benchmark = 'openai/gsm8k'
ds = datasets.load_dataset(benchmark, 'main')
for shots_num in [64, 128]: #[1, 4, 8, 16, 32]:
    for split in ['train', 'test']:
        ds_split = ds[split].map(lambda x: pick_shots(x, shots_num, benchmark, 'question', 'answer'), num_proc=16)
        ds_split.select_columns(['text']).push_to_hub('yananchen/gsm8k_shots_{}'.format(shots_num), split = split,  token='')

############################
benchmark = 'deepmind/aqua_rat'
ds = datasets.load_dataset(benchmark).map(lambda x: process_aqua_rat(x), num_proc=16)
ds['test'] = datasets.concatenate_datasets([ds['test'], ds['validation']])
ds['train'] = ds['train'].shuffle(seed=333).select(range(7500))
for shots_num in [64, 128]: #[1, 4, 8, 16, 32]:
    for split in ['train', 'test']:
        ds_split = ds[split].map(lambda x: pick_shots(x, shots_num, benchmark, 'question', 'answer'), num_proc=16)
        ds_split.select_columns(['text']).push_to_hub('yananchen/aqua_shots_{}'.format(shots_num), split = split,  token='')


######################
benchmark = 'lighteval/MATH'
ds = datasets.load_dataset(benchmark)
for shots_num in [64, 128]: #[1, 4, 8, 16, 32]:
    for split in ['train', 'test']:
        ds_split = ds[split].map(lambda x: pick_shots(x, shots_num, benchmark, 'problem', 'solution'), num_proc=16)
        ds_split.select_columns(['text']).push_to_hub('yananchen/math_shots_{}'.format(shots_num), split = split,  token='')


######################
benchmark = 'TIGER-Lab/MMLU-Pro'
ds = datasets.load_dataset(benchmark).map(lambda x: process_mmlu_pro(x), num_proc=16)
ds['train'] = ds['validation']

for shots_num in [1, 4]:
    for split in ['train', 'test']:
        ds_split = ds[split].map(lambda x: pick_shots(x, shots_num, benchmark, 'question', 'answer'), num_proc=16)
        ds_split.select_columns(['text']).push_to_hub('yananchen/mmlupro_shots_{}'.format(shots_num), split = split,  token='')

shots_num = -1
for split in ['train', 'test']:
    ds_split = ds[split].map(lambda x: pick_shots(x, shots_num, benchmark, 'question', 'answer'), num_proc=16)
    ds_split.select_columns(['text']).push_to_hub('yananchen/mmlupro_sft', split = split,  token='')





'''
benchmark = "hotpotqa/hotpot_qa"
ds = datasets.load_dataset(benchmark, 'fullwiki').map(lambda x: process_hotpotqa(x), num_proc=16)
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


benchmark = 'gsm8k'
ds = datasets.load_dataset("yananchen/{}_moa_weak_than_llama31".format(benchmark))
ds_icl = datasets.load_dataset("yananchen/{}_shots_8".format(benchmark))

assert ds['test'].num_rows == ds_icl['test'].num_rows

usage = []
ds_dic = {'text':[]}
for text_icl, text_moa in zip(ds_icl['test'], ds['test']):
    q = text_icl['text'].split('QUESTION:')[-1].split('SOLUTION:')[0].strip()
    assert q in text_moa['text']
    icl = text_icl['text'].split('<<<Examples ends')[0] + '<<<Examples ends'
    sys = text_icl['text'].split('Examples starts>>>')[0]
    question = text_icl['text'].split('<<<Examples ends')[-1].split('SOLUTION:')[0]
    solution = 'SOLUTION:' + text_icl['text'].split('<<<Examples ends')[-1].split('SOLUTION:')[-1]

    opinions = 'OPINIONS START' + text_moa['text'].split('OPINIONS START')[-1].split('OPINIONS END')[0] + 'OPINIONS END\n'
    ds_dic['text'].append(icl  + question + 
                '\nhere are some opinions from external committee for the given question above, separator is >>>, followed by the index of each opinion. \n' 
                 + opinions + '\nnow start write your solution below.\n' + solution)

    usage.append(len(tokenizer.encode(sys  + question + opinions +  solution)))

print(min(usage), max(usage), sum(usage)/len(usage))

# print(random.sample(ds_dic['text'], 1)[0])

datasets.Dataset.from_dict(ds_dic).push_to_hub('yananchen/{}_noicl_moa_phi3'.format(benchmark), 
                                    split = 'test', 
                                    token='')




benchmark = 'gsm8k'
ds = datasets.load_dataset("yananchen/{}_moa_weak_than_llama31".format(benchmark))


def rm_cot_in_annotation(example):
    example['text'] = example['text'].split('OPINIONS END')[0]+'OPINIONS END\nSOLUTION:' \
                            + '#### ' + \
                            example['text'].split('OPINIONS END')[-1].split('SOLUTION:')[-1].split('####')[-1]
    return example

ds_rm_cot = ds.map(lambda x: rm_cot_in_annotation(x), num_proc=16)
ds_rm_cot.push_to_hub('yananchen/{}_moa_weak_than_llama31_rm_cot_in_annotation'.format(benchmark), 
                                    token='')




