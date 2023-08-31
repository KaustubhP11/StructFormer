import torch
import os
import shutil
import re

def extract_tokens(num_tokens,num_nodes,num_headers,num_tokens_per_header,file):
    temp=len([token for word in file['title'][0] for token in word])
    if(temp):
        num_tokens += temp
        num_nodes += 1
        num_headers += 1
        num_tokens_per_header += temp

    temp=len([token for word in file['content'][0][0] for token in word])
    if(temp):
        num_tokens += temp
        num_nodes += 1

    if 'sublevels' in file:
        for dictt in file['sublevels']:
            a,b,c,d = extract_tokens(0,0,0,0,dictt)
            num_tokens += a
            num_nodes += b
            num_headers += c
            num_tokens_per_header += d

    return (num_tokens,num_nodes,num_headers,num_tokens_per_header)


ip_path='/data2/venkat/parser_code/dataset_pkl_all/'
op_path='/data2/venkat/parser_code/new_pklfiles_k_2k_4k/'

all_pkl_files = [name for name in os.listdir(ip_path) if name.endswith(".pkl")]
docs=[]
# c=0
file_counter=1
for pkl_file in all_pkl_files:  # for pkl_file in ['17741.pkl','18453.pkl']:
    with open (ip_path+pkl_file,'rb') as f:
        data=torch.load(f,encoding='utf-8')
        for file in data:   # one by one 10 files
            num_tokens,num_nodes,num_headers,num_tokens_per_header = extract_tokens(0,0,0,0,file)

            if(num_tokens>=2000 and num_tokens<=4000):
                docs.append(file)
                torch.save(docs,op_path+str(file_counter)+'.pkl')
                docs=[]
                # print(file_counter," file done")
                file_counter=file_counter+1
                if(file_counter%500 ==0):
                    print(file_counter)
                # c=c+1
                # print(c," done")

            # if(c==10):
            #     torch.save(docs,op_path+str(file_counter)+'.pkl')
            #     docs=[]
            #     file_counter=file_counter+1
            #     c=0
            #     print(file_counter," file done")

# with open('1.pkl','a') as t:
#     t.write(str(docs))
#     print(len(docs))

            # if(num_tokens>24000 or num_tokens<2000):
                # flag=0
                # break

            # with open('stats_from_pkl_3.txt','a') as t:
            #     t.write(pkl_file+" : ")
            #     t.write(str(num_tokens))
            #     t.write(',')
            #     t.write(str(num_nodes))
            #     t.write(',')
            #     t.write(str(num_headers))
            #     t.write(',')
            #     t.write(str(num_tokens_per_header))
            #     t.write('\n')
        # if(flag):
            # with open('pkl_files_2k_to_16k.txt','a') as t:
            #     t.write(pkl_file)
            #     t.write('\n')
            # shutil.copy('/data2/venkat/parser_code/dataset_pkl_all/'+pkl_file,'/data1/venkat/program/master/tarun/scratch_train/pkl_files_2k_16k')
        
# #print("max : ",max_tokens)   # 4553287


# '''Code to count number of tokens manually'''
# with open ('/data2/venkat/parser_code/dataset_pkl_all/18453.pkl','rb') as f:
#     data=torch.load(f,encoding='utf-8')
#     for filee in data:
#         # with open('18453_pkl_file.txt','a') as t:
#         #     t.write(str(file))
#         #     t.write('\n\n\n\n')
#         file0=str(filee)
#         file1=re.sub("\[","",file0)
#         file2=re.sub(']',"",file1)
#         file3=re.sub('\{','',file2)
#         file4=re.sub('}','',file3)
#         file5=re.sub(',','',file4)
#         file6=re.sub('title','',file5)
#         file7=re.sub('content','',file6)
#         file8=re.sub('sublevels','',file7)
#         file9=re.sub("'':",'',file8)
#         print(len(file9.split()))
