import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import torch,time
from tqdm import tqdm
import sys,os,re,random
from argparse import ArgumentParser, Namespace

from sugar_binding_predictor import sugar_binding_predictor


from pdb_n2n_input import data_preprocessing

parser = ArgumentParser()

parser.add_argument('--complex_pdb', type=str, default=None, help='.pdb file ')#'example_complex_list'
parser.add_argument('--complex_list', type=str, default=None, help='.pdb file list, each file can add a label after the file name seperated by a tab')
parser.add_argument('--site_txt', type=str, default=None, help='Processed minimum binding site input .txt file')
parser.add_argument('--site_list', type=str, default=None, help='Processed minimum binding site input .txt list, each file can add a label after the file name seperated by a tab')
parser.add_argument('--with_label', type=int, default=1, help='Label 1 or 0 of input file, if unknown set None')
parser.add_argument('--file_path', type=str, default='../example/test_pdb/', help='Directory of input file or file list ')
parser.add_argument('--out_dir', type=str, default='../', help='Directory where the outputs will be written to')
parser.add_argument('--model_dir', type=str, default='../model/param/', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--model', type=str, default=None, help='Model parameter for prediction, there are well-trained and general two options, in default both of them would be used ')
parser.add_argument('--output_CH_pi_record', type=int, default=1, help='Write CH-pi interaction info for each pdb file ')

args = parser.parse_args()


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def dataset_for_test_only(loc_list,label_list,list_limit=None):
    if isinstance(loc_list, list):
        pass
    else:
        loc_list=[loc_list]

    all_file_list=[]
    for n,loc in enumerate(loc_list):
        file_list = []
        label=label_list[n]
        for f in os.listdir(loc):
            if re.search('(.*)elicit_no_CH_pi_info\.txt', f):
                file_list.append([loc + f[:-24] + 'elicit_info.txt',label])
        if file_list == []:
            for f in os.listdir(loc):
                if re.search('(.*)elicit_info\.txt', f):
                    file_list.append([loc + f,label])
        #random.shuffle(file_list)
        if list_limit != None:
            limit = list_limit[n]
            file_list = file_list[:limit]
        count_file = len(file_list)
        print(str(count_file) + ' samples of binding site would be process \n')

        all_file_list.extend(file_list)
    #print(all_file_list)
    return all_file_list
def load_data_from_file(file_path):
    info_dict={}
    with open(file_path, 'r') as info_file:
        sample_info = info_file.readlines()
        info_file.close()
    for line in sample_info:
        if line != '\n':
            perline = line.split('\n')[0].split('\t')
            info_dict[perline[0]] = perline[1]
    info_dict['clean_interacting_list'] = eval(info_dict['clean_interacting_list'])
    return info_dict

def load_sample():
    output_CH_pi_record=args.output_CH_pi_record
    test_only_set=[]
    file_path=args.file_path
    if args.complex_pdb==args.complex_list==args.site_txt==args.site_list==None:
        print('No file is loading. Please use any  of --complex_pdb, --complex_list, --site_txt, --site_list. ')
        print('Running example ... ')
        args.complex_list='example_complex_list'
        #exit()
    if args.site_txt is not None:
        test_only_set.append([file_path+args.site_txt,args.with_label])
    if args.site_list is not None:
        with open(file_path+args.site_list,'r') as site_list_file:
            context=site_list_file.readlines()
            try:
                site_info=[[file_path+i.split('\t')[0],int(i.split('\t')[1].split('\n')[0])]for i in context]
            except IndexError:
                site_info = [[file_path + i.split('\n')[0].split(' ')[0], None] for i in context]
                args.with_label=None
            test_only_set.extend(site_info)
    if args.complex_pdb is not None:
        blockPrint()
        data_preprocessing(file_path,args.complex_pdb,output_CH_pi_record=output_CH_pi_record)
        enablePrint()
        test_only_set.append([file_path + args.complex_pdb[:-4] + '_elicit_info.txt', args.with_label])
    if args.complex_list is not None:
        print('\t Processing .pdb file...')
        with open(file_path+args.complex_list,'r') as complex_list_file:
            context=complex_list_file.readlines()
            try:
                complex_info=[[file_path+i.split('\t')[0],int(i.split('\t')[1].split('\n')[0])]for i in context]
            except IndexError:
                complex_info = [[file_path + i.split('\n')[0].split(' ')[0], None] for i in context]
                args.with_label=None
            for cp in tqdm(complex_info):
                complex_pdb_file=cp[0]
                blockPrint()
                data_preprocessing(file_path,complex_pdb_file[len(file_path):],output_CH_pi_record=output_CH_pi_record)
                enablePrint()
            test_only_set.extend([[i[0][:-4] + '_elicit_info.txt',i[1]] for i in complex_info])
    return test_only_set


def predict_binding(out_path='../'):
    select_param_loc =args.model_dir
    param_file_list=[]
    if args.model is None:
        param_file_list = ['ndv2_general.pt','ndv2_well.pt',]
    elif args.model=='well-trained':
        param_file_list = ['ndv2_well.pt']
    elif args.model=='general':
        param_file_list = ['ndv2_general.pt']
    print('\nModel to be run :',param_file_list)
    param_index = 0

    if args.with_label==None:
        test_result = pd.DataFrame(
            columns=['params_name', 'test_size', '1', '0',])
    else:
        test_result = pd.DataFrame(
            columns=['params_name', 'test_accuracy', 'test_size', '11', '10', '00', '01'])

    for select_param in param_file_list:
        param_index += 1
        model.load_state_dict(torch.load(select_param_loc + select_param))
        print('Running model ' + select_param + ' ' + str(param_index) + '/' + str(len(param_file_list)) + '... ')




        accuarcy_list = []
        accuarcy_file_list = []
        pred_result_list = []
        with torch.no_grad():
            for  i in tqdm(test_only_set):

                label = i[1]
                i = i[0]

                try:
                    info_dict = load_data_from_file(i)
                    blockPrint()
                    pred = model(info_dict)
                    enablePrint()
                except FileNotFoundError:
                    pred=torch.tensor([[0.00]])

                try:
                    accuarcy_list.append(str(label) + str(round(float(pred))))
                except ValueError:
                    continue
                accuarcy_file_list.append(i)
                pred_result_list.append(float(pred))



        accuarcy_dict = {ii: accuarcy_list.count(ii) for ii in accuarcy_list}
        if  args.with_label==None:
            for term in ['None1', 'None0']:
                try:
                    accuarcy_dict[term]
                except KeyError:
                    accuarcy_dict[term] = 0
            new_row = {'params_name': select_param[:-3],
                       'test_size': sum(accuarcy_dict.values()),
                       '1': accuarcy_dict['None1'], '0': accuarcy_dict['None0'],}
            print(accuarcy_dict, '')
            test_result= pd.concat([test_result, pd.DataFrame([new_row])], ignore_index=True)
        else:
            for term in ['11', '10', '01', '00']:
                try:
                    accuarcy_dict[term]
                except KeyError:
                    accuarcy_dict[term] = 0
            accuracy = (accuarcy_dict['11'] + accuarcy_dict['00']) / sum(accuarcy_dict.values())
            print(f' accuacry: {accuracy * 100:.2f}% in {sum(accuarcy_dict.values()):.0f}',
                  accuarcy_dict, '')

            new_row = {'params_name': select_param[:-3],
                       'test_accuracy': accuracy,
                       'test_size': sum(accuarcy_dict.values()),
                       '11': accuarcy_dict['11'], '10': accuarcy_dict['10'], '00': accuarcy_dict['00'],
                       '01': accuarcy_dict['01']}
            test_result= pd.concat([test_result, pd.DataFrame([new_row])], ignore_index=True)
        with open(out_path + select_param[:-3] + f'_test_{sum(accuarcy_dict.values())}_record.txt', "w") as f:
            for r1, r2, r3 in zip(accuarcy_list, accuarcy_file_list, pred_result_list):
                if 'None' in r1:
                    r1=r1[-1]
                f.write(str(r1) + '\t' + str(r3) + "\t" + str(r2).split('/')[-1].split('_elicit_info.txt')[0] +'\t'+select_param[:-3]+ "\n")

    pd.set_option('display.max_rows', 500)
    print(test_result)
    test_result.to_excel(out_path + f'/test_result_{sum(accuarcy_dict.values()):.0f}.xlsx')



if __name__ == '__main__':

    ### load model
    random.seed(2023)
    device = torch.device('cpu')
    deg = torch.tensor([0, 10989, 16001, 15933, 3905, 774, 169, 82, 79, 183, 17, 11, 39, 0, 1])
    model = sugar_binding_predictor(device=device, deg=deg, fragment_cluster=1, sugar_cluster=None)
    loss_func = torch.nn.L1Loss()

    print('\n\nprocessing sample...')
    test_only_set=load_sample()
    predict_binding(out_path=args.out_dir)

