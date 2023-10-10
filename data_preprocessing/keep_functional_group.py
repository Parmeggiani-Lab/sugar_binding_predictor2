import os,re,pymol,shutil
from pymol import cmd,stored
import operator


loc_ori=loc
loc=loc+'/inter_resi/'

file_list=[]
for f in os.listdir(loc):
    if re.search('(.*)\.pdb', f):
        file_list.append(f)
count_file=len(file_list)
print(str(count_file)+' residue files will elicit functional group')
#print(all_file)

if os.path.exists(loc_ori+'/functional_group')==False:
    os.mkdir(loc_ori+'/functional_group')

loc_out=loc_ori+'/functional_group/'

os.chmod(loc_out, 0o777)


with open(loc_ori+'CH_Pi_score.txt', 'r') as CH_pi_file:
    CH_pi_info_list = CH_pi_file.readlines()
    CH_pi_file.close()

CH_pi_info_list=[[eval(i.split('\t')[1]),i.split('\t')[2]] for i in CH_pi_info_list]

CH_pi_info_dict={}
for i in CH_pi_info_list:
    CH_pi_info_dict[i[1][:-4]]=i[0]
print(CH_pi_info_dict)


"""
atom to keep:
    1. sugar main ring + rest of sugar
    2. Chi-1 functional group of side-chain of N/O/S interacting amino acid (at least two heavy atom)
    3. N/O interacting main chain (alter residue name to GLY)
    4. aromatic sidechain with CH-pi score record
    
info to keep:
    1. atom coordinates
    2. atom type
    3. fragment type
    4. adjacent matrix
    5. bond type

"""
aa_keep_atom_dict={
    'SER':'OG+CB',
    'THR':'OG1+CB',
    'ASN':'OD1+ND2+CG',
    'GLN':'OE1+NE2+CD',
    'CYS':'CB+SG',
    'PRO':'CA+N+CD+CG+CB',
    'MET':'CG+SD',
    'PHE':'CG+CD1+CD2+CE1+CE2+CZ',
    'TYR':'CG+CD1+CD2+CE1+CE2+CZ+OH',
    'TRP':'CG+CD1+NE1+CE2+CD2+CE3+CZ3+CH2+CZ2',
    'ARG':'NH2+NH1+CZ',
    'HIS':'CG+ND1+CE1+NE2+CD2',
    'LYS':'NZ+CE',
    'ASP':'CG+OD1+OD2',
    'GLU':'CD+OE1+OE2',

}


for pdb in file_list[0:]:#for pdb in file_list[0:]:#

    print(file_list.index(pdb))
    print(pdb)


    cmd.load(loc+pdb)

    cmd.select('CH_pi_side', 'None')
    cmd.select('CH_pi_sugar', 'None')
    try:
        CH_pi_info=CH_pi_info_dict[pdb[:-9]]
        CH_pi_residue='+'.join([i.split('_')[0] for i in CH_pi_info])
        CH_pi_sugar='+'.join([i.split('_')[2] for i in CH_pi_info])
        #print(CH_pi_info)
        #print(CH_pi_residue,CH_pi_sugar)
        if CH_pi_residue !='':
            cmd.select('CH_pi_side','sidechain and i. '+CH_pi_residue+'')
            cmd.select('CH_pi_sugar', 'i. '+CH_pi_sugar+'')
    except KeyError:
        pass
    #if print(cmd.select('////PRO')!=0):
    #    print(pdb)


    cmd.select('sc_1','byres((not polymer around 3.5) and sidechain) ')
    cmd.select('sc_interacting','sc_1 and sidechain')
    cmd.select('bb_1','byres((not polymer around 3.5) and backbone) ')
    cmd.select('bb_interacting','bb_1 and backbone')

    cmd.remove('not ( (not polymer) or sc_interacting or bb_interacting or CH_pi_side) or hydrogen')
    #cmd.save(loc_out+pdb[:-8]+'elicit.pse')
    for aa in aa_keep_atom_dict:
        #print(aa,aa_keep_atom_dict[aa])
        cmd.remove('(sc_interacting or CH_pi_side) and ////'+aa+' and (not n. '+aa_keep_atom_dict[aa]+')')
    cmd.alter('bb_interacting','resn="GLY"')


    ##### keep only interacting sugar
    cmd.select('inter_sugar','byres(polymer around 3.5)')
    cmd.remove('(not polymer) and (not inter_sugar) and (not CH_pi_sugar)')

    cmd.set('pdb_conect_all','on')
    if cmd.select('not polymer')>0 and cmd.select('polymer')>0:
        cmd.save(loc_out+pdb[:-8]+'elicit.pdb')
    cmd.reinitialize()


for _,_,i in os.walk(loc_out):
    for file in i:
        os.chmod(loc_out+file, 0o777)