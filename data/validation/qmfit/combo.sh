set -x
F=scripts/qm/fit_water_qm.py
G='--vdw gvdw --free q=all;cov=all;radius=all;alpha=all;gvdw_sqrt_a=all;gvdw_sqrt_c6=all;gvdw_b=all --init gvdw_sqrt_a:HW=5;gvdw_sqrt_c6:HW=0.01;gvdw_b:HW=1'
W="total=1,elst=0.03,ind=0.03,exch_disp=0.03,nb3=10,force=0.1,dipole=1,polarizability=1,prior=0.01"
python $F fit rec_lj --weights "$W"
python $F fit rec_gvdw $G --weights "$W"
python $F fit rec_gvdw_nosapt $G --weights "total=1,nb3=10,force=0.1,dipole=1,polarizability=1,prior=0.01"
