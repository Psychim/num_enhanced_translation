python number_detector/extract_number.py --input_path ~/controller/hyp0$1.txt --output_path mt0$1hypnum.txt --mode parse
python number_detector/extract_number.py --input_path mt0$1hypnum.txt --output_path evalmt0$1.txt --reference_path mt0$1refnum.txt --mode evaluate --evaluate-mode EM
python number_detector/extract_number.py --input_path mt0$1hypnum.txt --output_path evalmt0$1.txt --reference_path mt0$1refnum.txt --mode evaluate --evaluate-mode value

