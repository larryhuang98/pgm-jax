set -x
F=scripts/qmfit/fit_water_qm.py
python $F fit probe_nb3x10 --w "total=1,nb3=10,dipole=1,polarizability=1,prior=0.01"
python $F fit probe_nomono --w "total=1,nb3=1,prior=0.01"
python $F fit probe_nb3only --free "radius=all;alpha=all" --w "total=0,nb3=1,prior=0.001"
