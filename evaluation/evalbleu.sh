ref_dir=/home/data_ti5_c/shicm/data/nist_zh-en_1.34m/test
sacrebleu $ref_dir/mt0$1.ref0 $ref_dir/mt0$1.ref1 $ref_dir/mt0$1.ref2 $ref_dir/mt0$1.ref3 -i hyp0$1.txt
