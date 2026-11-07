from task1.main import run as task1
from task2.main import parser, run as task2
from task3.main import run as task3


def main():
    cli = parser()
    cli.add_argument('task', choices=['all', 'task1', 'task2', 'task3'])
    args = cli.parse_args()
    tasks = {'task1': task1, 'task2': task2, 'task3': task3}
    for name, run in tasks.items():
        if args.task in (name, 'all'):
            run(args)


if __name__ == '__main__':
    main()
