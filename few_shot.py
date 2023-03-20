import json
import requests
import os.path as osp
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import opt as opts
# 并行推理
import deepspeed
import torch.distributed as dist
import os
import torch.nn as nn

'''调用API'''
def query(payload='',parameters=None,options={'use_cache': False}):
    # 基于GPT-Neo
    API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    body = {"inputs":payload,'parameters':parameters,'options':options}
    response = requests.request("POST", API_URL, headers=headers, data= json.dumps(body))
    try:
      response.raise_for_status()
    except requests.exceptions.HTTPError:
        return "Error:"+" ".join(response.json()['error'])
    else:
      return response.json()[0]['generated_text']
    

'''本地部署推理'''
def load_gpt_neo_model():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    
    return tokenizer, model

def load_gpt_j_model():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    nn.Dataparallel(model, device_ids=[0, 1])

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    # os.environ['WORLD_SIZE'] = '2'

    # 初始化全局分组 指定world_size 因为mp_size需要与world_size相同
    # group = dist.init_process_group(backend='nccl', 
    #                                 init_method='env://', 
    #                                 world_size=1, 
    #                                 rank=int(os.environ('RANK')))

    # 添加并行化代码 可将模型分布式地置于多个GPU上 防止显存不足
    # ds_engine = deepspeed.init_inference(model, 
    #                                      mp_size=1, # 使用GPU数量
    #                                      dtype=torch.float16, # The data type to use.
    #                                      checkpoint=None,
    #                                     #  group = group,
    #                                      replace_with_kernel_inject=True, # replace the model with the kernel injector
    #                                      )
    # model = ds_engine.module

    return tokenizer, model

def load_gpt_neox_model():
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    # 添加并行化代码 可将模型分布式地置于多个GPU上 防止显存不足
    ds_engine = deepspeed.init_inference(model, 
                                         mp_size=3, # 使用GPU数量
                                         dtype=torch.float16, # The data type to use.
                                         checkpoint=None,
                                         replace_with_kernel_inject=True, # replace the model with the kernel injector
                                         )
    model = ds_engine.module

    return tokenizer, model

class KeywordsCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 生成出现关键词则停止
        if input_ids[0][-1] in self.keywords:
            return True
        return False

def query_from_local(tokenizer, model, stop_criteria, prompt):
    
    tokens = tokenizer(prompt, return_tensors="pt")
    # tokens = tokenizer(prompt)
    input_ids = tokens.input_ids.to(device)
    # attention_mask = tokens.attention_mask

    gen_tokens = model.generate(
        input_ids,
        # attention_mask,
        do_sample=True,
        temperature=0.1,
        # 相当于MaxLengthCriteria
        max_length=len(prompt) + 30,
        # 添加KeywordsCriteria
        stopping_criteria=StoppingCriteriaList([stop_criteria]), 
    )
    
    gen_text = tokenizer.batch_decode(gen_tokens)

    return gen_text[0]
    # return gen_text['generated_text']

# 多进程时必须添加该语句 确保只有主进程会执行该代码 而不会在子进程重复执行
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)

    opt = opts.parse_opt()
    name = opt.name
    dataset = opt.dataset

    # HuggingFace API Token
    API_TOKEN = "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"

    parameters = {
    'max_new_tokens':30,  # number of generated tokens
    'temperature': 0.1,   # controlling the randomness of generations
    'end_sequence': "###" # stopping sequence for generation
    }

    if name == 'neo':
        tokenizer, model = load_gpt_neo_model()
    elif name == 'j':
        tokenizer, model = load_gpt_j_model()
    elif name == 'neox':
        tokenizer, model = load_gpt_neox_model()
    else:
        raise ValueError('Model not exist!')

    stop_words = ['###']
    stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
    stop_criteria = KeywordsCriteria(stop_ids)


    '''加载待转换问题及Prompt'''
    prompt = '' # few-shot prompt
    with open('./prompt.txt', 'r') as f:
        prompt = f.read()   
    # print(prompt) 

    path_v2 = './vqa_v2_train_yn_question.csv'
    path_v2_output = f'./vqa_v2_train_wh_question_with_{name}.csv'
    path_v2_unexpected = f'./vqa_v2_train_unexpected_question_with_{name}.csv'

    path_cpv2 = './vqacp_v2_train_yn_question.csv'
    path_cpv2_output = f'./vqacp_v2_train_wh_question_with_{name}.csv'
    path_cpv2_unexpected = f'./vqacp_v2_train_unexpected_question_with_{name}.csv'

    if dataset == 'vqav2':
        path_csv = path_v2
        path_output = path_v2_output
        path_unexpected = path_v2_unexpected
    elif dataset == 'vqacpv2':
        path_csv = path_cpv2
        path_output = path_cpv2_output
        path_unexpected = path_cpv2_unexpected

    ques = pd.read_csv(path_csv, sep='\t')
    ques = ques[['question_id', 'question', 'multiple_choice_answer']]

    print(ques)

    '''转换'''
    # 输出文件
    with open(path_output, 'w+') as f:
        # 异常输出文件
        with open(path_unexpected, 'w+') as uf:
            for i, yn in ques.iterrows():
                # if i == 10:
                #     break
                cur_prompt = prompt + yn[1] + ' ' + yn[2]
                try:
                    '''在线'''
                    # data = query(cur_prompt,parameters)
                    '''离线'''
                    data = query_from_local(tokenizer, model, stop_criteria, cur_prompt)
                    # print(data)

                    res = data.split('\n')
                    # print(res)
                    # 若res为空 取倒数第三行
                    res = res[-2] if res[-2] else res[-3]
                    # print(res)
                    if res.startswith('Original'):
                        raise Exception('Started with Original')

                    # 写入文件
                    f.write(f'{yn[0]} : {res}')
                    f.write('\n')
                    f.flush()

                except Exception as e:
                    # 可用于处理 IndexError: list index out of range
                    # 记录意外情况 下一次迭代再生成
                    # (question_id, info)
                    uf.write(f'{yn[0]} : {e}')
                    uf.write('\n')
                    uf.write(f'{data}')
                    uf.write('\n')
                    uf.flush()
                    pass

                # 继续执行
                continue
                
    print('Done!')
    
