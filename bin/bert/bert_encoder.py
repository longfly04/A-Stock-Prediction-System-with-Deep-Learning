'''
概述：
    使用bert_as_service服务，编码新闻信息，异步处理文本并进行编码，server作为单独进程，client可以多进程请求server

Server API:

    Argument		Type		Default		Description
    model_dir		str		Required		folder path of the pre-trained BERT model.
    max_seq_len		int		25		maximum length of sequence, longer sequence will be trimmed on the right side. Set it to NONE for dynamically using the longest sequence in a (mini)batch.
    num_worker		int		1		number of (GPU/CPU) worker runs BERT model, each works in a separate process.
    max_batch_size		int		256		maximum number of sequences handled by each worker, larger batch will be partitioned into small batches.
    port		int		5555		port for pushing data from client to server
    port_out		int		5556		port for publishing results from server to client
    gpu_memory_fraction		float		0.5		the fraction of the overall amount of memory that each GPU should be allocated per worker
    cpu		bool		False		run on CPU instead of GPU

Client API

    Argument	Type	Default	Description
    ip	str	localhost	IP address of the server
    port	int	5555	port for pushing data from client to server, must be consistent with the server side config
    port_out	int	5556	port for publishing results from server to client, must be consistent with the server side config
    show_server_config	bool	False	whether to show server configs when first connected
    timeout	int	-1	set the timeout (milliseconds) for receive operation on the client

'''

from multiprocessing import Process,Event
import os
import sys
import time
from bert_serving.client import BertClient
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
import pandas as pd 
import numpy as np 

sys.path.append('C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\A-Stock-Prediction-System-with-GAN-and-DRL')
model_path = "C:\\Users\\longf.DESKTOP-7QSFE46\\GitHub\\chinese-bert_chinese_wwm_L-12_H-768_A-12\\publish"

def run_service(args,):
    bs = BertServer(args)
    bs.start()
    print('[Server] Server is running.\n')
    return bs
    
def run_clinet(context, bs):# client 等待server 的ready事件
    print('[Client] Client is running.\n')
    bs.is_ready.wait(30)
    if bs.is_ready:
        with BertClient() as bc:
            output = bc.encode(context)
            # print(output)
        return output
    else:
        return None

def encode_minutes(data, 
                    server_args, 
                    content_column='content', 
                    code_column='code', 
                    msg_len=10, 
                    save=True, 
                    save_filename=None
                    ):
    '''
    概述：
        对分钟级新闻进行编码
    参数：
        data：新闻数据dataframe
        content_column：新闻内容列
        code_column:编码列
        len:一次性发送给server的新闻条数
    输出：
        字符串形式的list编码
    '''
    text_len = msg_len
    if code_column not in data.columns:
        data[code_column] = pd.Series().astype(object)
    
    bs = run_service(server_args) #启动服务

    for i in range(int(len(data)/text_len)):
        if type(data[code_column].iloc[i*text_len]) != str and bs.is_ready: # code列中的数据为Nan时（encoder返回的结果以str形式保存）
            print("[encoder] Start encode row %d to row %d" % (i*text_len, (i+1)*text_len-1))
            text = data[content_column].iloc[i*text_len:(i+1) * text_len]
            coder = run_clinet(list(text), bs).reshape((text_len, -1)) # 启动客户端，传入文本和server的状态
            try:
                for j in range(text_len):
                    data.at[i*text_len+j, code_column] = coder.tolist()[j] # 这里用at而不能用iloc或者loc
            except Exception as e:
                print(e)
                os.system('bert-serving-terminate -port 5555')
                time.sleep(16)
            if save and i%32 == 0 or i == int(len(data)/text_len)-1:# 为了减少读写次数 每32次请求保存一次数据
                data.to_csv(save_filename) 
        elif bs.is_ready: # 遇到已经编码的行
            print("[encoder] Row %d to row %d is already encoded." %(i*text_len, (i+1)*text_len-1))
        else: # 服务已经停止
            os.system('bert-serving-terminate -port 5555') # 发送终止命令给server
            time.sleep(10)
            break

if __name__ == '__main__':
    server_args = get_args_parser().parse_args(['-model_dir', model_path,
                                            '-port', '5555',
                                            '-port_out', '5556',
                                            '-max_seq_len', '512', # 最大序列长度动态调整
                                            '-num_worker', '1', # 1个worker接收序列，并将其分配给GPU
                                            ])
    filename = "dataset\\News-2019-07-28-to-2019-01-30.csv"
    codename = filename.replace('News','News_with_code')
    if os.path.exists(codename):
        data = pd.read_csv(codename)
    else:
        data = pd.read_csv(filename)
    try:
        data = data.drop(columns=[x for x in data.columns if x.startswith('Unnamed: ')]) # 删除奇怪的列
    except Exception as e:
        print(e)
        
    encode_minutes(data, server_args=server_args, save_filename=codename)
    
    

    


