# import nltk
# from nltk.corpus import stopwords
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from telnetlib import SE
from typing import Dict, List, Tuple, Optional, Any
import torch, random, numpy as np
from datasets import list_metrics, load_metric
from transformers import LongformerForMaskedLM, LongformerTokenizer, TrainingArguments, HfArgumentParser
# from transformers import Trainer
from trainer import SMTrainer as Trainer
from torch.utils.data.dataset import Dataset, IterableDataset
import os, natsort, math, tqdm
from dataclasses import dataclass
# import tracemalloc

class SMDataset(IterableDataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,        
        dir_path: str,
        instances_per_file=100,
        eval = False
    ):
        #print("\n*********SMDataset init*********")
        super(SMDataset).__init__()
        assert os.path.isdir(dir_path), f"Input dir path {dir_path} not found"
        self.files = natsort.natsorted([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name)) and name.endswith('.pkl')])
        self.instances_per_file = instances_per_file
        self.dir_path = dir_path
        self.eval = eval
    def __len__(self):
        #print("\n*********SMDataset len*********",len(self.files)*self.instances_per_file)
        return len(self.files)*self.instances_per_file

    def __iter__(self):
        #print("\n*********SMDataset iter*********")
        self.multiprocess_loading()
        for i, file in enumerate(self.files):
            tfile = torch.load(os.path.join(self.dir_path, file))
            for ind in range(self.instances_per_file):
                if self.eval:
                    yield((tfile[ind], self.eval))
                else:
                    yield((tfile[ind], self.eval))


    def multiprocess_loading(self):
        #print("\n*********SMDataset multiprocess_loading*********")
        worker_info = torch.utils.data.get_worker_info()
        #print("********worker_info : ",worker_info)
        if worker_info is not None:  # multi-process data loading
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            self.files = self.files[iter_start : iter_end]

#    def __getitem__(self, i):
#        document = torch.load(os.path.join(self.dir_path, self.files[i//self.instances_per_file]))[i%self.instances_per_file]
#        return document, -1


@dataclass
class DataCollatorForSM:
    """
    Data collator used for model batch input. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    """
    tokenizer:Any = None
    pad_to_multiple_of: Optional[int] = None
    max_len:int = 16384
    max_node:int = 60
    vocab:Dict = None
    #sw_nltk = None#stop words for reducing global tokens 
    

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        #print("*********DataCollatorForSM call*********")
        #print("*********features*********",features)
        pad_to_multiple_of = self.pad_to_multiple_of
        max_len = self.max_len
        self.vocab = self.tokenizer.get_vocab()

        flat_example_list = [self.traverse_and_flatten(each_example, []) for each_example, i in features]
        
        #( [{input_ids : [], attention_mask:[], global_attention_mask:[], labels:[]}, {}, {}, ...])
        batch_inputs = [self.insert_node_ids(flat_list=item, node_token_id=self.vocab["</s>"], max_len=max_len, max_node=self.max_node, add_node = False) for item in flat_example_list]
        
        #club batches together
        batch = batch_inputs[0]
        
        #find max len in a batch
        max_len_b = max(list(map(lambda a: len(a["input_ids"][0]), batch_inputs)))
        
        keys = list(batch.keys())
        for i in range(1, len(batch_inputs)):
            for k in keys:
                batch[k].append(batch_inputs[i][k][0])#note batch_inputs[i]["key"] is [[]] and not []
        
        if False:#features[-1][-1]:
            batch.pop("labels")
            keys.remove("labels")
        #do pad
        # print(max_len, "#########")
        max_len_b = ((max_len_b // pad_to_multiple_of) + 1) * pad_to_multiple_of if max_len_b % pad_to_multiple_of != 0 else max_len_b     
        # print(max_len, "***********", max_len_b)                 
        if max_len_b > max_len:#drop last
            for k in keys:
                batch[k] = drop_batch(batch[k], max_len)
        else:#pad
            for k in keys:   
                if k == "labels":
                    fill_val = -100
                elif k == "input_ids":
                    fill_val = self.vocab["<pad>"]#<pad> token in tokenizer vocab 
                elif k in ["attention_mask", "global_attention_mask"]:
                    fill_val = 0              

                batch[k] = fill_batch(batch[k], max_len, fill_val)
        
        # for k in keys:
        #     for i in range(len(batch[k])):
        #         print(len(batch[k]), len(batch[k][i]))
        #         for j in range(len(batch[k][i])):
        #             print(type(batch[k][i][j]), int)
        #             if type(batch[k][i][j]) != int:
        #                 print(i, j , k)
        #                 exit()
        # print(batch)
        # exit()
        # for k in keys:
        #     print(k,type( np.array(batch[k])))
        # return
        # create tensors
        # for k in keys:
        #     print(torch.tensor(batch[k], dtype=torch.long).shape, k)
        for k in keys:
            batch[k] = torch.tensor(batch[k], dtype=torch.long)
        return batch
    def traverse_and_flatten(self, node, flat):
        '''
        do a dfs to extract nodes and make it into list
        '''
        #print("************node************",node)
        for key in node.keys():
            if key == "title":
                flat.append(("t", node[key][0]))
                #print("************node************",node[key][0])
                # print("title", len(node[key][0]))
                # if len(node[key][0]) in {0,1}:
                #     print(node[key][0])

            elif key == "content":
                flat.append(("c", node[key][0][0]))
                # print("content", len(node[key][0][0]))
                # if len(node[key][0][0]) in {0,1}:
                #     print(node[key][0][0])
            elif key == "sublevels":       
                for each_node in node["sublevels"]:
                    flat = self.traverse_and_flatten(each_node, flat)
       
        return flat    
    def insert_node_ids(self, flat_list, node_token_id, max_len, max_node, add_node = False):
        '''
            flat_list = [(t for title, [[1564,16],[84,987],46...]), (c for content, [[46],[4,64],[646,5644 ], ...]), (), (), .....]
            inserts node id if add_node is False then concatenated 
            else : do only concatenate
        '''
        input_ids = []
        token_type_ids = []
        global_attention_mask = []
        attention_mask = []
        max_len_reached = 0
        node_id = 0
        if add_node:
            #then insert node tokens and accordingly obtain node segment id
            pass
        else:
            for node in flat_list:#concatenate all input ids, add global attention
                # [('t', [[5087, 8190], [627], [45696], [17272, 21553], [463]]), ('c', [[42274], [12], [6486, 1342], [1640], ....])]
                # for word in node[1]:
                #     for tok in word:
                #         print("********************tok : ",tok)
                
                # print("node : ",node)
                num_of_tokens = len([tok for word in node[1] for tok in word])
                # print("number of tokens : ",num_of_tokens)
                max_len_reached += num_of_tokens
                if max_len_reached > max_len or node_id > max_node:
                    break
                input_ids += node[1]
                
                if node[0] == "t":                    
                    #depending on node's t or c insert token type id
                    #global_attention_mask += [1]*num_of_tokens#0 for local 1 for global
                    global_attention_mask += [0]*num_of_tokens#0 for local 1 for global
                    #token_type_ids += [node_id]*num_of_tokens
                    #input_ids += [[node_token_id]]#append node token
                    #token_type_ids += [node_id]#append node token id
                    #global_attention_mask += [1]#make node token global
                    #node_id += 1
                else:
                    global_attention_mask += [0]*num_of_tokens#0 for local 1 for global
                    #token_type_ids += [node_id]*num_of_tokens                

        #generate labels for a task
        input_ids, labels = self.do_mask_for_mlm(input_ids, node_token_id)
        
        # print(len(input_ids), len(labels), len(global_attention_mask), len(attention_mask))

        input_ids = [tok for word in input_ids for tok in word]
        labels = [tok for word in labels for tok in word]
        attention_mask = [1]*len(input_ids)
        # print(len(input_ids), len(labels), len(global_attention_mask), len(attention_mask), len(token_type_ids), token_type_ids[-1])
        # print((input_ids), (labels), (global_attention_mask), (attention_mask), (token_type_ids))
        # 1/0

        return {
            "input_ids" : [input_ids],
            "global_attention_mask" : [global_attention_mask],
            #"token_type_ids" : [token_type_ids],
            "attention_mask" : [attention_mask], 
            "labels" : [labels]
            }

    def do_mask_for_mlm(self, input_id, node_token_id):
        #currently no whole word masking
        #assuming <mask> is not ignored in labels using -100 also cls and sep token is not present for -100 in labels and not here pad is not considered therefore no need to maintain condition for pad
        mask_id = self.vocab["<mask>"]
        ignore_id = -100
        maked_input_id = []
        labels = []#[ignore_id]*number_of_tok
        indices = list(range(0, len(input_id)))
        random.shuffle(indices)
        selected_indices = random.sample(indices, int(len(indices)*0.15))
        for i in range(len(input_id)):
            if i in selected_indices and input_id[i] != node_token_id:
                if random.random() < 0.8:#mask the token
                    maked_input_id.append([mask_id]*len(input_id[i]))
                    labels.append(input_id[i])
                else:
                    if random.random() < 0.5:
                        #if word has more than one tokens then for this step choose last word piece or middle at random because we dont want to take care of word piece starting or not straing with space
                        if len(input_id[i]) > 1:
                            maked_input_id.append(input_id[i][:-1] + [random.randint(5, len(self.vocab)-1)])
                        else:
                            maked_input_id.append([random.randint(5, len(self.vocab)-1)])
                        labels.append([ignore_id]*len(input_id[i]))
                    else:
                        maked_input_id.append(input_id[i])
                        labels.append([ignore_id]*len(input_id[i]))
            else:
                maked_input_id.append(input_id[i])
                labels.append([ignore_id]*len(input_id[i]))
        
        return [maked_input_id, labels]

def fill_batch(x, max_len, fill_val):
    for i in range(len(x)):
        x[i] += [fill_val]*(max_len - len(x[i]))
    return x
def drop_batch(x, max_len):
    for i in range(len(x)):
        x[i] = x[i][0:max_len]
    return x
# def compute_accuracy(pred, true):
#     max_len = max(list(map(lambda a:len(a), pred)))
#     pred, true = fill_batch(pred, max_len, 0), fill_batch(true, max_len, -100)

#     pred, true = np.array(pred), np.array(true)
    
#     acc = 100*np.sum(np.multiply((pred == true),(true != -100)))/np.sum(true != -100)
#     return acc
def compute_predictions(test_dataset, data_collator, model):
    for inputs in tqdm.tqdm(test_dataset):        
        batch_prediction, batch_ref = [], []
        model_inputs = data_collator([inputs])
        model_inputs_d = {}
        for k in model_inputs.keys():
            if k != "labels":
                model_inputs_d[k] = model_inputs[k].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model_predictions = model(**model_inputs_d)
        _, prediction = torch.max(model_predictions.logits, -1)
        torch.cuda.empty_cache()
        batch_prediction += prediction.tolist()
        batch_ref += model_inputs["labels"].tolist()
    return [batch_prediction, batch_ref]

def train(path_test, path_train):
    parser = HfArgumentParser(TrainingArguments)
    #print("\nTrainingArguments*******",TrainingArguments)
    #print("\nparser********",parser)
    training_args = parser.parse_args_into_dataclasses(look_for_args_file=False)
    training_args[0].remove_unused_columns = False
    training_args[0].per_device_train_batch_size=2
    training_args[0].per_device_eval_batch_size=1
    training_args[0].num_train_epochs=2
    training_args[0].dataloader_num_workers=1
    training_args[0].save_steps=10000
    training_args[0].logging_steps=1000
    training_args[0].fp16 = True
    training_args[0].gradient_checkpointing=True

    # training_args[0].prediction_loss_only = False
    # print(training_args[0])
    sw_nltk = None#stopwords.words('english')

    # model = LongformerForMaskedLM.from_pretrained('../../../pretrained_16392_65k_mlm')#, gradient_checkpointing=True)
    # tokenizer = LongformerTokenizer.from_pretrained('../../../pretrained_16392_65k_mlm')
    model = LongformerForMaskedLM.from_pretrained('/data1/venkat/program/mod_longformer-base-4096-init', gradient_checkpointing=True)
    tokenizer = LongformerTokenizer.from_pretrained('/data1/venkat/program/mod_longformer-base-4096-init')
    test_dataset = SMDataset(dir_path=path_test, instances_per_file=10, eval = True)
    train_dataset = SMDataset(dir_path=path_train, instances_per_file=10)
    #print("\ntest train initialised*************************",training_args)
    data_collator = DataCollatorForSM(tokenizer=tokenizer, pad_to_multiple_of = 512, max_len = 4096, max_node=1)
    torch.cuda.empty_cache()
    #print("\ndata initialised*************************")
    #print bpc
    trainer = Trainer(model=model, args=training_args[0], data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=test_dataset)
    torch.cuda.empty_cache()
    #print("\ntrainer initialised*************************")
    #eval_loss = trainer.evaluate()
    
    #print("epoch number ", 0, "evaluation loss", eval_loss["eval_loss"], "eval bpc", eval_loss["eval_loss"]/math.log(2), "accuracy ", eval_loss["eval_accuracy"])
    torch.cuda.empty_cache()
    train_loss = trainer.train()
    print("epoch number", training_args[0].num_train_epochs, "trianing loss", train_loss.training_loss, "train bpc", train_loss.training_loss/math.log(2))
    torch.cuda.empty_cache()
    eval_loss = trainer.evaluate()
    print("epoch number", training_args[0].num_train_epochs, "evaluation loss", eval_loss["eval_loss"], "eval bpc", eval_loss["eval_loss"]/math.log(2),  "accuracy ", eval_loss["eval_accuracy"])

    trainer.save_model(training_args[0].output_dir)

def train_test(path_test, path_train):
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses(look_for_args_file=True)
    model = LongformerForMaskedLM.from_pretrained('../../../allen_ai_longformer-base-4096', gradient_checkpointing=True)
    tokenizer = LongformerTokenizer.from_pretrained('../../../allen_ai_longformer-base-4096')
    test_dataset = SMDataset(dir_path=path_test, instances_per_file=10)
    train_dataset = SMDataset(dir_path=path_train, instances_per_file=10)

    data_collator = DataCollatorForSM(tokenizer=tokenizer, pad_to_multiple_of = 512, max_len = 12000)
    
    x = []
    k = 4
    for i in train_dataset:
        x.append(i)
        k -= 1
        if k == 0:
            break
    print(len(x))
    print(data_collator(x))
    exit()
    training_args[0].remove_unused_columns = False
    print(training_args)

    trainer = Trainer(model=model, args=training_args[0], data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=test_dataset)

    trainer.train()

#train("/data2/venkat/parser_code/dataset_pkl_all/test", "/data2/venkat/parser_code/dataset_pkl_all/train")
#train("../../../../dataset/subset/test", "../../../../dataset/subset/train")
#  /data1/venkat/program/master/tarun/scratch_train/pkl_files_2k_16k_train
# tracemalloc.start()
train("/data2/venkat/parser_code/new_pklfiles_2k_12k_test", "/data2/venkat/parser_code/new_pklfiles_2k_12k_train")
# train("/data2/venkat/parser_code/new_pklfiles_2k_12k_test", "/data2/venkat/parser_code/new_pklfiles_2k_12k_train")
# [current,peak=tracemalloc.get_traced_memory()
# print("current : ",current)
# print("peak : ",peak)]
# tracemalloc.stop()
