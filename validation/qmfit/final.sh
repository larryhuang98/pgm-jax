set -x
python scripts/qmfit/collect_water_qm.py
F=scripts/qmfit/fit_water_qm.py
python $F baseline
python $F fit lj_total --free "lj_rmin_half=OW;lj_sqrt_eps=OW" --w "total=1,prior=0.01"
python $F fit all_total --w "total=1,nb3=1,dipole=1,polarizability=1,prior=0.01"
python $F fit all_total_F --w "total=1,nb3=1,force=0.1,dipole=1,polarizability=1,prior=0.01"
for s in 0.01 0.03 0.1 0.3 1; do
python $F fit all_sapt_$s --w "total=1,elst=$s,ind=$s,exch_disp=$s,nb3=1,dipole=1,polarizability=1,prior=0.01"
done
python $F fit all_sapt_only --w "total=0,elst=1,ind=1,exch_disp=1,nb3=1,dipole=1,polarizability=1,prior=0.01"
G='--vdw gvdw --free q=all;cov=all;radius=all;alpha=all;gvdw_sqrt_a=all;gvdw_sqrt_c6=all;gvdw_b=all --init gvdw_sqrt_a:HW=5;gvdw_sqrt_c6:HW=0.01;gvdw_b:HW=1'
python $F fit gvdw_total $G --w "total=1,nb3=1,dipole=1,polarizability=1,prior=0.01"
python $F fit gvdw_sapt_0.3 $G --w "total=1,elst=0.3,ind=0.3,exch_disp=0.3,nb3=1,dipole=1,polarizability=1,prior=0.01"
python $F fit gvdw_sapt_only $G --w "total=0,elst=1,ind=1,exch_disp=1,nb3=1,dipole=1,polarizability=1,prior=0.01"
python $F summary lj_total,all_total,all_total_F,all_sapt_0.01,all_sapt_0.03,all_sapt_0.1,all_sapt_0.3,all_sapt_1,all_sapt_only,gvdw_total,gvdw_sapt_0.3,gvdw_sapt_only
