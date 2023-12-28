##################################
#
# PyRosetta-depended script
#
##################################
import ast
import sys,os
sys.path.remove('/home/tm21372/Rosetta/PyRosetta/setup')
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
blockPrint()

import pyrosetta; pyrosetta.init()
import os
from pyrosetta import *
from rosetta.core.select import residue_selector as selections
from rosetta.core import select
from rosetta.core.pack.task import operation
import numpy as np
import math
from pyrosetta.rosetta.core.chemical import ResidueProperty

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/math.pi

def get_center(ring_coords_list):
    x_sum = 0
    y_sum = 0
    z_sum = 0
    count = 0
    for atom in ring_coords_list:
        x_sum += atom[0]
        y_sum += atom[1]
        z_sum += atom[2]
        count += 1

    vec1=ring_coords_list[0]-ring_coords_list[1]
    vec2 = ring_coords_list[0] - ring_coords_list[2]
    vec_vert=np.cross(vec1,vec2)
    return [x_sum / count, y_sum / count, z_sum / count],vec_vert

def motif_info_standardize(v1):
    return [(float(v1[0])-2)/(3.5-2),(float(v1[1])+180)/360,(float(v1[2])+180)/360,(float(v1[3])+180)/360]

def run_dist(v1, v2):
    try:
        dist= np.sqrt(np.sum((v1 - v2) ** 2))
    except TypeError:
        v1=np.array(v1)
        v2 = np.array(v2)
        dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist

def vec_pm_select(input_vec):
    output_list=[]
    for i,yes_no in enumerate(input_vec):
        if yes_no==1:
            output_list.append(str(i+1))
    #print(output_list)
    list_translated='+'.join(output_list)
    command='select target,i. '+list_translated
    print(command)
    return output_list

def angle(a0,a1,a2):
    vec1=[a1[0]-a0[0],a1[1]-a0[1],a1[2]-a0[2]]
    vec2 = [a2[0] - a0[0], a2[1] - a0[1], a2[0] - a0[2]]

    cos=(vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2])/(math.sqrt(vec1[0]*vec1[0]+vec1[1]*vec1[1]+vec1[2]*vec1[2])*math.sqrt(vec2[0]*vec2[0]+vec2[1]*vec2[1]+vec2[2]*vec2[2]))
    angle=math.acos(cos)*180/math.pi
    if angle >360:
        angle=360-angle
    return angle

def dihedral(a0,a1,a3,a4):
    vec0=[a1[0]-a0[0],a1[1]-a0[1],a1[2]-a0[2]]
    vec1 = [a3[0] - a0[0], a3[1] - a0[1], a3[2] - a0[2]]
    vec2 = [a4[0] - a1[0], a4[1] - a1[1], a4[2] - a1[2]]

    vec_pl1=[vec1[1]*vec0[2]-vec1[2]*vec0[1],vec1[2]*vec0[0]-vec1[0]*vec0[2],vec1[0]*vec0[1]-vec1[1]*vec0[0]]
    vec_pl2 = [vec0[1] * vec2[2] - vec0[2] * vec2[1], vec0[2] * vec2[0] - vec0[0] * vec2[2], vec0[0] * vec2[1] - vec0[1] * vec2[0]]

    try:
        cos = (vec_pl1[0] * vec_pl2[0] + vec_pl1[1] * vec_pl2[1] + vec_pl1[2] * vec_pl2[2]) / (math.sqrt(vec_pl1[0] * vec_pl1[0] + vec_pl1[1] * vec_pl1[1] + vec_pl1[2] * vec_pl1[2]) * math.sqrt(vec_pl2[0] * vec_pl2[0] + vec_pl2[1] * vec_pl2[1] + vec_pl2[2] * vec_pl2[2]))
        sup_angle=math.acos(cos)*180/math.pi
        if sup_angle >360:
            sup_angle=sup_angle-360
    except ZeroDivisionError:
        sup_angle=180
    return 180-sup_angle


known_sugar_ligand_dict={
    'SGN':[['C1','C2','C3','C4','C5'],['H1','H2','H3','H4','H5']]
}
options = """
-ignore_unrecognized_res
-include_sugars
-auto_detect_glycan_connections
-maintain_links 
-alternate_3_letter_codes pdb_sugar
-write_glycan_pdb_codes
-ignore_zero_occupancy false 
-load_PDB_components 1
-no_fconfig
"""

init(" ".join(options.split('\n')))



def find_CH_pi_aromatic(pose):
    sugar_selector = selections.ResiduePropertySelector()
    sugar_property = ResidueProperty.CARBOHYDRATE
    sugar_selector.set_property(sugar_property)
    sugar_vec = sugar_selector.apply(pose)
    sugar_list = vec_pm_select(sugar_vec)
    print(sugar_list)
    if sugar_list==[]:
        sugar_selector = selections.ResiduePropertySelector()
        sugar_property = ResidueProperty.LIGAND
        sugar_selector.set_property(sugar_property)
        sugar_vec = sugar_selector.apply(pose)
        sugar_list = vec_pm_select(sugar_vec)
        print('\n')
        print('ligand list:',sugar_list)
        print('\n')
    # find CH of sugar
    CH_list = []
    for sugar in sugar_list:
        # print(pose.residue(int(sugar)))
        resinfo = str(pose.residue(int(sugar))).split('\n')[6].split()
        if resinfo[0] != 'Ring': #### recognize as ligand
            if (str(pose.residue(int(sugar))).split('\n')[1].split()[1]) in known_sugar_ligand_dict:
                sugar_ligand=(str(pose.residue(int(sugar))).split('\n')[1].split()[1])
                print('recogize '+sugar_ligand+' : '+str(known_sugar_ligand_dict[sugar_ligand]))
                C_of_CH_list=known_sugar_ligand_dict[sugar_ligand][0]
                H_of_CH_list=known_sugar_ligand_dict[sugar_ligand][1]
                for i,Hi in zip(C_of_CH_list,H_of_CH_list):  # sugarid_C_H
                    C_id = pose.residue(int(sugar)).atom_index(i)
                    # print(C_id)
                    H_id = pose.residue(int(sugar)).atom_index(Hi)
                    # print(str(sugar)+'_'+str(C_id)+'_'+str(H_id))
                    C_coord = pose.residue(int(sugar)).xyz(C_id)
                    # print('\tC_coord:',C_coord)
                    H_coord = pose.residue(int(sugar)).xyz(H_id)
                    # print('\tH_coord:', H_coord)
                    CH_vec = C_coord - H_coord
                    #pose.residue(int(sugar)).name()
                    CH_list.append(
                        [str(pose.pdb_info().number(int(sugar))) + '_' + i, sugar_ligand,
                         [float(C_coord[0]), float(C_coord[1]), float(C_coord[2])], np.array(CH_vec),
                         [float(H_coord[0]), float(H_coord[1]), float(H_coord[2])]])

                #print(CH_list)

            else:
                raise TypeError
        else: #### recognize as normal sugar
            C_of_CH_list = [i for i in resinfo[2:] if i[0] == 'C']

            print(C_of_CH_list)

            for i in C_of_CH_list:  # sugarid_C_H
                C_id = pose.residue(int(sugar)).atom_index(i)
                #print(C_id)
                try:
                    H_id = [i for i in pose.residue(int(sugar)).get_hydrogens_bonded_to_ring_atom(C_id)][0]
                    # print(str(sugar)+'_'+str(C_id)+'_'+str(H_id))
                    C_coord = pose.residue(int(sugar)).xyz(C_id)
                    #print('\tC_coord:',C_coord)
                    H_coord = pose.residue(int(sugar)).xyz(H_id)
                    #print('\tH_coord:', H_coord)
                    CH_vec = C_coord - H_coord
                    CH_list.append([str(pose.pdb_info().number(int(sugar))) + '_' + i, pose.residue(int(sugar)).name(),
                                    [float(C_coord[0]), float(C_coord[1]), float(C_coord[2])], np.array(CH_vec),[float(H_coord[0]), float(H_coord[1]), float(H_coord[2])]])
                except IndexError: ## no H on this ring-C
                    continue
            #print(CH_list)

    nbr_selector = selections.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(sugar_selector)
    nbr_selector.set_include_focus_in_subset(False)
    neigbh_vec = nbr_selector.apply(pose)
    neigbh_list = vec_pm_select(neigbh_vec)
    aromatic_list = [int(i) for i in neigbh_list if pose.residue(int(i)).is_aromatic() == 1]
    #print(aromatic_list)

    # find aromatic ring massive center and vertical vector
    aromatic_ring_list = []
    for i in aromatic_list:
        resinfo = str(pose.residue(int(i))).split('\n')[6].split()
        if resinfo[0] != 'Ring':
            print(pose.residue(i).name())
            raise TypeError
        # print(resinfo)
        ring_coords_list = []
        for j in resinfo[2:]:
            # print(pose.residue(i).xyz(j))
            ring_coords_list.append(pose.residue(i).xyz(j))
        center = get_center(ring_coords_list)
        aromatic_ring_list.append([i, pose.residue(i).name(), center[0], center[1]])
        if pose.residue(i).name() == 'TRP':
            resinfo = str(pose.residue(int(i))).split('\n')[7].split()
            # print(resinfo)
            if resinfo[0] != 'Ring':
                raise TypeError
            ring_coords_list = []
            for j in resinfo[2:]:
                # print(pose.residue(i).xyz(j))
                ring_coords_list.append(pose.residue(i).xyz(j))
            center = get_center(ring_coords_list)
            aromatic_ring_list.append([i, str(pose.residue(i).name()) + '_B', center[0], center[1]])
    #print(aromatic_ring_list)

    # calculate 3 term

    CH_pi_info_list = []
    for aromatic_ring in aromatic_ring_list:
        for CH_pair in CH_list:
            chi2_ori= float(pose.residue(int(aromatic_ring[0])).chi(2))
            chi2=float("{:.2f}".format(chi2_ori))
            C_X_dist = run_dist(CH_pair[2], aromatic_ring[2])
            H_X_dist = run_dist(CH_pair[4], aromatic_ring[2])
            if H_X_dist<C_X_dist:
                CH_vert_angle = angle_between(CH_pair[3], aromatic_ring[3])
                if CH_vert_angle > 90:
                    CH_vert_angle = 180 - CH_vert_angle
                CX_vec = np.array(aromatic_ring[2]) - np.array(CH_pair[2])
                CX_vert_angle = angle_between(CX_vec, aromatic_ring[3])
                Cp_X_dist = C_X_dist * np.sin(CX_vert_angle * np.pi / 180)
                CH_pi_info_list.append([C_X_dist, CH_vert_angle, Cp_X_dist, pose.pdb_info().number(aromatic_ring[0]), aromatic_ring[1], CH_pair[0],chi2,
                                        pose.pdb_info().name()])
                #print(aromatic_ring[0],pose.pdb_info().number(aromatic_ring[0]))
    return CH_pi_info_list

def pose_CH_pi_score(CH_pi_aromatic_list):
    CH_pi_score=0
    CH_pi_record_list=[]
    for i in CH_pi_aromatic_list:

        #print(i)
        if i[0]<4.5: #C-X distance cutoff=4.5A
            if i[1]<40: #CH-Pi angle cutoff=40 degree
                if i[4]=='HIS' or i[4]=='TRP': #for His, Trp_A
                    if i[2]<1.6: #C-projection distance cutoff=1.6A
                        print(i)
                        CH_pi_score+=1
                        CH_pi_record_list.append(str(i[3])+'_'+i[4]+'_'+i[5]+'_'+str(i[6]))
                elif i[4]=='PHE' or i[4]=='TRP_B' or i[4]=='TYR': #for Phe, Trp_B, Tyr
                    if i[2]<2: #C-projection distance cutoff=2A
                        print(i)
                        CH_pi_score += 1
                        if i[4]=='TRP_B':
                            CH_pi_record_list.append(str(i[3]) + '_' + 'TRB' + '_' + i[5]+'_'+str(i[6]))
                        else:
                            CH_pi_record_list.append(str(i[3]) + '_' + i[4] + '_' + i[5] + '_' + str(i[6]))
    print('CH_pi_score :',CH_pi_score)
    return CH_pi_score,CH_pi_record_list

def measure_CH_pi(loc,each_file,out_record=1):
    blockPrint()
    pose_ori = pose_from_pdb(loc+each_file)
    pose = Pose()
    pose.assign(pose_ori)

    CH_pi_aromatic_list=find_CH_pi_aromatic(pose)
    result=pose_CH_pi_score(CH_pi_aromatic_list)
    CH_pi_score = result[0]
    CH_pi_record= result[1]

    if out_record==1:
        f = open(loc + 'CH_Pi_score.txt', 'a')
        f.write(str(CH_pi_score)+'\t'+str(CH_pi_record)+'\t'+each_file.split('/')[-1]+'\t'+each_file+'\n')
        f.close()
    return CH_pi_record

def main():
    loc='../example/test_pdb/'
    pdb_files = [file for file in os.listdir(loc) if file[-4:] == '.pdb']
    for pdb in pdb_files:
        measure_CH_pi(loc,pdb)


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()