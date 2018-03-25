import sys

def main():
    binary = True if sys.argv[1] == 'b' else False
    test_x_filepath = sys.argv[2]
    test_y_filepath = sys.argv[3]
    kaggle_filepath = sys.argv[4]

    with open(test_x_filepath) as test_x:
        with open(test_y_filepath) as test_y:
            with open(kaggle_filepath, 'w') as kaggle:
                print('x,y', file=kaggle)
                for x, y in zip(test_x, test_y):
                    x = x.rstrip('\n')
                    if binary:
                        y = y.rstrip('\n')
                    else:
                        y = y.rstrip('\n')
                        y = str(float(y))
                    print(x + ',' + y, file=kaggle)


if __name__ == '__main__':
    main()