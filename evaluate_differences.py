from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description='Find different lines.')

parser.add_argument('-i', default='./num2text-p8-v7/num2text_num_p8_v7.txt', help='input file')
parser.add_argument('-o', default='./num2text-p8-v7/num2text_txt_p8_v7.txt',  help='output file')
parser.add_argument('-p', default='predictions.txt', help='prediction file')

args = parser.parse_args()

input_path = args.i
output_path = args.o
predicted_path = args.p

diff_input_path = 'diff_input.txt'
diff_output_path = 'diff_output.txt'
diff_predict_path = 'diff_predict.txt'

differences = {}
nl = 0

with open(output_path, 'r') as output:
    with open(predicted_path, 'r') as predicted:
        while True:
            output_line = next(output, None)
            predicted_line = next(predicted, None)
            if output_line != predicted_line:
                if not output_line or not predicted_line:
                    print(nl, output_line, predicted_line)
                    break
                diff = (None, output_line, predicted_line)
                differences[nl] = diff

                if nl % 100 == 0:
                    print('\r%d' % nl, end='')
            elif not output_line:
                break
            nl += 1

print("differences", len(differences))
nl = 0
differences_compact = {}
print (differences.keys())
with open(input_path, 'r') as input:
    for line in input:
        if nl in differences.keys():
            i = 0
            for n in range(len(differences[nl][1])):
                i = n
                if differences[nl][1][i] != differences[nl][2][i]:
                    break
            i -= i // 5
            differences_compact[nl] = line, differences[nl][1][i:], differences[nl][2][i:]
            differences.pop(nl, None)
            print('\r%d' % nl, end='')
        nl += 1
print()

print(differences_compact.keys())

with open(diff_input_path, 'w') as in_f:
    with open(diff_output_path, 'w') as out_f:
        with open(diff_predict_path, 'w') as pred_f:
            for _, item in differences_compact.items():
                in_f.write(item[0])
                out_f.write(item[1])
                pred_f.write(item[2])


