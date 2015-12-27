import optparse
import pandas as pd

PATH = 'data/gifts.csv'

def parse_arg():
    parser = optparse.OptionParser('usage%prog')
    parser.add_option('-S', dest='source', type='string', help='source path of csv file')
    (options, args) = parser.parse_args()
    return options


def sample(path):
    ds = pd.read_csv(path)
    l = len(ds)
    num = input('Found {0} instances, please specify how sample numbers:\n'.format(l))
    num = int(num)
    if num > l or num < 0:
        print('[-] Error, sample number could only be in [{0}, {1}]:\n'.format(0, l))
        exit(0)
    ds = ds.sample(n=num)
    name = input('Please specify the name of output csv file:\n')
    ds.to_csv(name)


def main():
    arg = parse_arg()
    if arg.source is None:
        sample(PATH)
    else:
        sample(path)


if __name__ == '__main__':
    main()