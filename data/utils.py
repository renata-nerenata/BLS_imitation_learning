STANDART = ['A', 'T', 'G', 'C']


def get_tenzor_encoding_input(puzzle, elements=STANDART):
    tenzor = []
    max_len = 0
    for i in puzzle:
        if len(i) > max_len:
            max_len = len(i)
    for i in elements:
        line = []
        for j in puzzle:
            string = []
            for k in j:
                if k == i:
                    string.append(1)
                else:
                    string.append(0)
            while len(string) < max_len:
                string.append(0)
            line.append(string)
        tenzor.append(line)
    return tenzor


def get_tenzor_encoding_output(puzzle):
    tenzor = []
    max_len = 0
    for i in puzzle:
        if len(i) > max_len:
            max_len = len(i)
    line = []
    for j in puzzle:
        string = []
        for k in j:
            if k == '-':
                string.append(1)
            else:
                string.append(0)
        while len(string) < max_len:
            string.append(0)
        line.append(string)
    tenzor.append(line)
    return tenzor


def apply_step_shift(list_, sm):
    max_len = 0
    for i in list_:
        if len(i) > max_len:
            max_len = len(i)
    elem = list_[sm[0]]
    new_elem = ''
    for i in range(len(elem)):
        if i != sm[1]:
            new_elem += elem[i]
        else:
            new_elem += '-'
            new_elem += elem[i]
    max_len = len(new_elem)

    list_[sm[0]] = new_elem
    for i in range(len(list_)):
        if len(list_[i]) < max_len:
            list_[i] += '-'
    return list_


def get_reverse_encoding(matr, elements=STANDART):
    if len(matr) > len(elements):
        print("Wrong!")
        return

    max_len = len(matr[0][0])
    list_ = []
    for i in range(len(matr[0])):
        new_elem = ''
        for j in range(max_len):
            if int(matr[0][i][j]) == 1:
                new_elem += elements[0]
            elif int(matr[1][i][j]) == 1:
                new_elem += elements[1]
            elif int(matr[2][i][j]) == 1:
                new_elem += elements[2]
            elif int(matr[3][i][j]) == 1:
                new_elem += elements[3]
            else:
                new_elem += '-'
        list_.append(new_elem)
    return list_


def get_len_for_input(list_):
    return len(max(list_, key=len))

