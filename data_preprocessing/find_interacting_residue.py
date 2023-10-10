import os,re,pymol,shutil,json
from pymol import cmd,stored
import operator





ch_pi_dict={}
ch_pi_record_dict={}

#print(ch_pi_dict)

loc=loc.replace('\\','/')
loc=loc+'/'

file_list=[]
for f in os.listdir(loc):
    if re.search('(.*)\.pdb', f):
        file_list.append(f)
count_file=len(file_list)
print(str(count_file)+' files in under clearance')
#print(all_file)

if os.path.exists(loc+'/inter_resi')==False:
    os.mkdir(loc+'/inter_resi')
os.chmod(loc+'/inter_resi', 0o777)
out_loc=loc+'/inter_resi/'

#cmd.load(loc + file_list[0],'ori')

with open(loc+'CH_Pi_score.txt', 'r') as CH_pi_file:
    CH_pi_info_list = CH_pi_file.readlines()
    CH_pi_file.close()

CH_pi_info_list=[[eval(i.split('\t')[1]),i.split('\t')[2]] for i in CH_pi_info_list]

CH_pi_info_dict={}
for i in CH_pi_info_list:
    CH_pi_info_dict[i[1][:-4]]=i[0]
print(CH_pi_info_dict)


to_align=0
for pdb in file_list[0:]:#['5XC2_51_poly_1.pdb']:#
    print(file_list.index(pdb))
    cmd.load(loc + pdb)
    cmd.remove('hydrogen')
    cmd.remove("(not (alt +'A') )and ( polymer)")
    cmd.remove('/////OXT')
    cmd.alter('all',"alt=''")
    try:
        CH_pi_info=CH_pi_info_dict[pdb[:-4]]
        list_of_ch_pi=[]
        for i in CH_pi_info:
            if i.split('_')[1]=='TRB':
                list_of_ch_pi.append('////' + 'TRP' + '`' + str(int(i.split('_')[0])-0))
            else:
                list_of_ch_pi.append('////' + i.split('_')[1] + '`' + str(int(i.split('_')[0])-0))


        CH_pi_residue=' or '.join(list_of_ch_pi)
    except KeyError:
        CH_pi_residue=''
    print('\tCH-pi residue',CH_pi_residue)

    if to_align=='1':
        cmd.load(loc + file_list[0], 'ori_align')
        cmd.align(pdb[:4]+' and not polymer','ori_align and not polymer')
        cmd.remove('ori_align')

    cmd.select('atom_NOS','symbol O+N+S')
    cmd.select('polar_charged','(byres( ((not polymer) and '+pdb[:-4]+') around 3.5) and atom_NO) and (////ARG or ////HIS or ////LYS or ////ASP or ////GLU or ////Ser or ////THR or ////ASN or ////GLN or ////TYR)')



    if CH_pi_residue!='':
        cmd.select('aromatic',CH_pi_residue+' and ((not polymer) around 10)')
        take_chi_pi_record = '1'
    else:
        cmd.select('aromatic', 'None')
        take_chi_pi_record = '0'

    cmd.remove('not (not polymer or polar_charged or aromatic)')
    #cmd.remove('polymer') ########only sugar modify
    resi_num=cmd.select('ca','/////CA')
    if resi_num>0: ########only sugar modify
        cmd.save(out_loc+pdb[:-4]+'_resi.pdb')
    if take_chi_pi_record=='1':
        if cmd.select('ca1','i. +'+CH_pi_residue)>0:
            with open(out_loc+pdb[:-4]+'_resi.pdb','a') as resi_file:
                resi_file.write('\n'+' '.join(CH_pi_info_dict[pdb[:-4]]))
                resi_file.close()
    cmd.reinitialize()

print(out_loc)


generate_align_pse_file=0
if generate_align_pse_file==1:
    file_list_resi=[]
    for f in os.listdir(out_loc):
        if re.search('(.*)_resi.pdb', f):
            file_list_resi.append(f)

    cmd.load(out_loc + file_list_resi[0],file_list_resi[0][:-9])
    cmd.select('sugar','(not polymer)')
    cmd.select('aromatic','None')
    for resi_pdb in file_list_resi[1:]:
        cmd.load(out_loc + resi_pdb,resi_pdb[:-9])
        cmd.align(resi_pdb[:-9],'sugar')
        cmd.hide('everything', resi_pdb[:-9])
        try:
            CH_pi_info=CH_pi_info_dict[resi_pdb[:-9]]
            CH_pi_residue='+'.join([i.split('_')[0] for i in CH_pi_info])
            if CH_pi_residue !='':
                cmd.select('aromatic','aromatic or ('+resi_pdb[:-9]+' and i. '+CH_pi_residue+')')
        except KeyError:
            pass


    cmd.zoom()

    cmd.select('hydrophobic','////ALA or ////VAL or ////ILE or ////LEU or ////MET')
    cmd.select('charged','////ARG or ////HIS or ////LYS or ////ASP or ////GLU')
    #cmd.select('aromatic','////PHE or ////TYR or ////TRP')
    cmd.select('polar','////SER or ////THR or ////ASN or ////GLN')
    cmd.select('special','////GLY or ////PRO or ////CYS')

    cmd.select('C_','symbol C')
    cmd.color('wheat','aromatic and C_')
    cmd.color('grey','polar and C_')
    cmd.color('teal','charged and C_')

    cmd.hide('everything', 'not sugar')
    cmd.hide('line', 'hydrogen')
    cmd.show('line', 'polymer')
    cmd.scene('overall', 'store')

    cmd.hide('everything', 'not sugar')
    cmd.hide('line', 'hydrogen')
    cmd.show('line', 'polar and sidechain')
    cmd.scene('polar', 'store')

    cmd.hide('everything', 'not sugar')
    cmd.hide('line', 'hydrogen')
    cmd.show('line', 'charged and sidechain')
    cmd.scene('charged', 'store')

    cmd.hide('everything', 'not sugar')
    cmd.hide('line', 'hydrogen')
    cmd.show('line', 'charged or polar')
    cmd.scene('charged_polar', 'store')

    cmd.hide('everything', 'not sugar')
    cmd.hide('line', 'hydrogen')
    cmd.show('line', 'aromatic and sidechain')
    cmd.scene('aromatic', 'store')


    sugar='GLC'

    cmd.save(out_loc+sugar+'_3.5A_'+str(count_file)+'_.pse')
    cmd.reinitialize()


for _,_,i in os.walk(out_loc):
    for file in i:
        os.chmod(out_loc+file, 0o777)