from get_args import get_args


def main():
    args = get_args()
    if args.model == "transformer":
        prediction = pred(args.puzzle)
    if args.model == "FCN":
        prediction = pred(args.puzzle)


if __name__ == "__main__":
    main()
