import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import torch,time
import sys,os,re,random

from sugar_binding_predictor import sugar_binding_predictor


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
def load_data_from_file(file_path,ch_pi_percent=None):
    info_dict={}

    if ch_pi_percent!= None:
        if random.random()<=ch_pi_percent:
            with open(file_path, 'r') as info_file:
                sample_info = info_file.readlines()
                info_file.close()
            info_dict['open_ch_pi_count']=1
        else:
            #print(file_path[:-15])
            with open(file_path[:-15]+'elicit_no_CH_pi_info.txt', 'r') as info_file:
                sample_info = info_file.readlines()
                info_file.close()
            info_dict['open_ch_pi_count'] = 0
        info_dict['bad_train_file_count'] = 1
    else:
        with open(file_path, 'r') as info_file:
            sample_info = info_file.readlines()
            info_file.close()
    for line in sample_info:
        if line != '\n':
            perline = line.split('\n')[0].split('\t')
            info_dict[perline[0]] = perline[1]
    info_dict['clean_interacting_list'] = eval(info_dict['clean_interacting_list'])
    return info_dict


def predict_binding(out_path='../'):
    ch_pi_option=0
    select_param_loc = '../model/param/'
    param_file_list = []
    for f in os.listdir(select_param_loc):
        if re.search('(.*).pt', f):
            param_file_list.append(f)
    print(param_file_list)
    param_index = 0

    test_reult = pd.DataFrame(
        columns=['params_name', 'test_set', 'test_loss', 'test_accuracy', 'test_size', '11', '10', '00', '01'])

    for select_param in param_file_list[0:]:
        param_index += 1
        model.load_state_dict(torch.load(select_param_loc + select_param))
        print('loading ' + select_param + ' ' + str(param_index) + '/' + str(len(param_file_list)) + '... ')

        print('Test model ...')
        if multi_test_set == []:
            multi_test_set.append(test_only_set)

        for test_set_id, each_test_set in enumerate(multi_test_set):
            print(multi_test_name[test_set_id])
            test_only_set = each_test_set
            test_loss = 0
            test_error = 0
            accuarcy_list = []
            accuarcy_file_list = []
            pred_result_list = []
            not_find_file = 0
            with torch.no_grad():
                for index, i in enumerate(test_only_set):
                    # print(i)
                    label = i[1]
                    i = i[0]

                    # try:
                    if label == 1:
                        info_dict = load_data_from_file(i)
                    else:
                        # try:
                        info_dict = load_data_from_file(i, ch_pi_percent=ch_pi_option)
                        # except FileNotFoundError:
                        #    info_dict = load_data_from_file(i)

                    blockPrint()
                    pred = model(info_dict)
                    enablePrint()
                    # except FileNotFoundError:
                    #    pred=torch.tensor([0],dtype=torch.float32)
                    #    not_find_file+=1
                    loss = loss_func(pred, torch.tensor([[int(label)]], device=device))

                    if not (torch.isinf(loss) or torch.isnan(loss)):
                        test_loss += loss.detach().cpu().item()
                        # print(str(param_index)+'/'+str(len(param_file_list))+"<<<<<<<" +str(index)+' /'+str(len(test_only_set))+': y = '+str(label)+', pred =', round(float(pred)), pred)
                        accuarcy_list.append(str(label) + str(round(float(pred))))
                        accuarcy_file_list.append(i)
                        pred_result_list.append(float(pred))
                    else:
                        print('\ttest:', i, loss)
                        test_error += 1

                test_loss = (test_loss / (len(test_only_set) + 1 - test_error))
            print(f"Test Loss: {test_loss:.6f}")

            accuarcy_dict = {ii: accuarcy_list.count(ii) for ii in accuarcy_list}
            for term in ['11', '10', '01', '00']:
                try:
                    accuarcy_dict[term]
                except KeyError:
                    accuarcy_dict[term] = 0
            accuracy = (accuarcy_dict['11'] + accuarcy_dict['00']) / sum(accuarcy_dict.values())
            print(f' accuacry: {accuracy * 100:.2f}% in {sum(accuarcy_dict.values()):.0f}',
                  accuarcy_dict, '')

            new_row = {'params_name': select_param[:-3],
                       'test_set': multi_test_name[test_set_id],
                       'test_loss': test_loss,
                       'test_accuracy': accuracy,
                       'test_size': sum(accuarcy_dict.values()),
                       '11': accuarcy_dict['11'], '10': accuarcy_dict['10'], '00': accuarcy_dict['00'],
                       '01': accuarcy_dict['01']}
            test_reult = test_reult.append(new_row, ignore_index=1)
            with open(out_path + select_param[:-3] + f'_test_{test_loss:.2f}_record.txt', "a") as f:
                f.write(select_param[:-3] + '\t' + str(accuarcy_dict) + '\t' + f'{test_loss:.2f}' + '\t' + str(
                    accuracy) + "\n")
                for r1, r2, r3 in zip(accuarcy_list, accuarcy_file_list, pred_result_list):
                    f.write(
                        str(r1) + '\t' + str(r3) + "\t" + str(r2).split('/')[-1].split('_elicit_info.txt')[0] + "\n")

    pd.set_option('display.max_rows', 500)
    print(test_reult)
    test_reult.to_excel(out_path + f'/test_result_{test_loss:.2f}.xlsx')



if __name__ == '__main__':

    ### load model
    random.seed(2023)
    device = torch.device('cpu')
    deg = torch.tensor([0, 10989, 16001, 15933, 3905, 774, 169, 82, 79, 183, 17, 11, 39, 0, 1])
    model = sugar_binding_predictor(device=device, deg=deg, fragment_cluster=1, sugar_cluster=None)
    loss_func = torch.nn.L1Loss()

    #### load test set and name
    multi_test_set = []
    multi_test_name = []
    multi_test_set.append(dataset_for_test_only(['../example/xylose/n_XYS_native/input/'], [1]))
    multi_test_name.append('XYS_native')

    ### run predictor
    predict_binding(out_path='../')

