import sys
from typing import Tuple, List, Set, Dict


class BoolVector:
    def __init__(self, vector: List[bool]):
        self.vector = vector


char_to_keyword = {
    '(': "opening brack",
    ')': "closing brack",
    'O': "null",
    'I': "one",
    '!': "inversion",
    'V': "disjunction",
    '&': "conjunction",
    '^': "XOR",
    '>': "implication",
    '=': "equivalence",
    '|': "Sheffer stroke",
}


def inversion_func(bool_vector: BoolVector, lst_len: int) -> BoolVector:
    assert len(bool_vector.vector) == lst_len, "Несовпадение длины списка bool_lst со значением аргумента lst_len"
    return BoolVector([not bl for bl in bool_vector.vector])


def disjunction_func(a: BoolVector, b: BoolVector, lst_len: int) -> BoolVector:
    assert len(a.vector) == len(b.vector), "Несовпадение длины списка bool_lst_a с длиной списка bool_lst_b"
    assert len(a.vector) == lst_len, "Несовпадение длины списков со значением аргумента lst_len"
    return BoolVector([bl_a or bl_b for bl_a, bl_b in zip(a.vector, b.vector)])


def conjunction_func(a: BoolVector, b: BoolVector, lst_len: int) -> BoolVector:
    assert len(a.vector) == len(b.vector), "Несовпадение длины списка bool_lst_a с длиной списка bool_lst_b"
    assert len(a.vector) == lst_len, "Несовпадение длины списков со значением аргумента lst_len"
    return BoolVector([bl_a and bl_b for bl_a, bl_b in zip(a.vector, b.vector)])


funcs_table = {
    "null":           (0,
                       lambda lst_len: BoolVector([False] * lst_len)
                       ),
    "one":            (0,
                       lambda lst_len: BoolVector([True] * lst_len)
                       ),
    "inversion":      (1, inversion_func),
    "disjunction":    (2, disjunction_func),
    "conjunction":    (2, conjunction_func),
    "XOR":            (2,
                       lambda a, b, lst_len:
                        disjunction_func(
                            conjunction_func(a, inversion_func(b, lst_len), lst_len),
                            conjunction_func(inversion_func(a, lst_len), b, lst_len),
                            lst_len
                        )
                       ),
    "implication":    (2, lambda a, b, lst_len:
                        disjunction_func(
                            inversion_func(a, lst_len),
                            b,
                            lst_len
                        )
                       ),
    "equivalence":    (2, lambda a, b, lst_len:
                        disjunction_func(
                            conjunction_func(inversion_func(a, lst_len), inversion_func(b, lst_len), lst_len),
                            conjunction_func(a, b, lst_len),
                            lst_len
                        )
                       ),
    "Sheffer stroke": (2, lambda a, b, lst_len:
                        inversion_func(conjunction_func(a, b, lst_len), lst_len)
                       ),
}


priorities = {
    "null":           0,
    "one":            0,
    "inversion":      1,
    "conjunction":    2,
    "disjunction":    3,
    "XOR":            3,
    "implication":    4,
    "Sheffer stroke": 4,
    "equivalence":    5
}


def split_bool_func(bool_func: str) -> Tuple[Set[str], List]:
    vars = set()
    # processed_func = ["opening brack"]
    pre_proc_func = []
    var_buffer = []
    for ch in bool_func:
        assert ch not in ['[', ']'], "Нельзя использовать символы '[', ']' при задании логических функций"
        if ch in char_to_keyword:
            if var_buffer:
                var = ''.join(var_buffer)
                vars.add(var)
                var_buffer = []
                pre_proc_func.append(var)
            pre_proc_func.append(char_to_keyword[ch])
        else:
            var_buffer.append(ch)
    if var_buffer:
        var = ''.join(var_buffer)
        vars.add(var)
        pre_proc_func.append(var)

    def recursion(idx: int, brack_level: int) -> Tuple[List, int]:
        lst = []
        while idx < len(pre_proc_func) and pre_proc_func[idx] != "closing brack":
            if pre_proc_func[idx] == "opening brack":
                sub_lst, closing_brack_idx = recursion(idx + 1, brack_level + 1)
                lst.append(sub_lst)
                idx = closing_brack_idx + 1
            else:
                lst.append(pre_proc_func[idx])
                idx += 1
        if not brack_level and idx != len(pre_proc_func):
            raise Exception("Неправильное задание ЛФ! (Лишние закрывающие скобки)")
        if idx == len(pre_proc_func) and brack_level:
            raise Exception("Неправильное задание ЛФ! (Не закрыта скобка)")
        return lst, idx

    processed_func, _ = recursion(0, 0)

    return vars, processed_func


def calc_bool_func(processed_func: List | BoolVector | str, var_values: Dict[str, BoolVector], lst_len: int) -> BoolVector:
    if isinstance(processed_func, BoolVector):
        return processed_func

    if isinstance(processed_func, str):
        assert processed_func in var_values, "Неправильное задание ЛФ! (Оператору в качестве операнда был передан оператор)"
        return var_values[processed_func]

    func_idxes = []
    for i, el in enumerate(processed_func):
        if not isinstance(el, list) and el in funcs_table:
            func_idxes.append(i)
    func_idxes.sort(key=lambda idx: priorities[processed_func[idx]])

    for j in range(len(func_idxes)):
        idx = func_idxes[j]
        operands_n, func = funcs_table[processed_func[idx]]
        try:
            if operands_n == 0:
                func_res = func(lst_len)
                processed_func[idx] = func_res
            elif operands_n == 1:
                a = calc_bool_func(processed_func[idx + 1], var_values, lst_len)
                func_res = func(a, lst_len)
                processed_func = processed_func[:idx] + [func_res] + processed_func[idx + 2:]
            elif operands_n == 2:
                a = calc_bool_func(processed_func[idx - 1], var_values, lst_len)
                b = calc_bool_func(processed_func[idx + 1], var_values, lst_len)
                func_res = func(a, b, lst_len)
                processed_func = processed_func[:idx - 1] + [func_res] + processed_func[idx + 2:]
        except IndexError:
            raise Exception("Неправильное задание ЛФ! (Не хватает операндов для оператора)")

        for k in range(j + 1, len(func_idxes)):
            if func_idxes[k] > idx:
                func_idxes[k] -= operands_n
    assert len(processed_func) == 1, "Неправильное задание ЛФ! (наличие лишней переменной или оператора)"
    return calc_bool_func(processed_func[0], var_values, lst_len)


def main() -> None:
    assert len(sys.argv) == 3, ("Неправильный вызов программы! Правильный вызов:\n"
    "python main.py <входной файл с расширением> <выходной файл с расширением>")

    with open(sys.argv[1], 'r', encoding='utf-8') as input_file:
        with open(sys.argv[2], 'w', encoding='utf-8') as output_file:
            for line in input_file:
                try:
                    in_str = line.strip()
                    func_str = ''.join(in_str.split())
                    assert func_str, "Пустая строка!"
                    print("ЛФ:", func_str, file=output_file)
                    vars_set, processed_func = split_bool_func(func_str)
                    # print(vars_set, processed_func)
                    vars_lst = list(vars_set)
                    vars_lst.sort()
                    all_vectors_len = 2 ** len(vars_set)
                    var_values = {var_name: BoolVector([False] * all_vectors_len) for var_name in vars_lst}
                    for i in range(all_vectors_len):
                        digit = i
                        for j in range(len(vars_lst)):
                            var_values[vars_lst[-j - 1]].vector[i] = bool(digit % 2)
                            digit >>= 1

                    F = calc_bool_func(processed_func, var_values, all_vectors_len)

                    print("Найденные переменные:", *vars_lst, file=output_file)
                    print(*vars_lst, '|', "F", sep='\t', file=output_file)
                    print('-' * len(vars_lst) * 4, '+', '-' * 7, sep='', file=output_file)
                    for i in range(all_vectors_len):
                        print(*[int(var_values[vars_lst[j]].vector[i]) for j in range(len(vars_lst))],
                              '|', int(F.vector[i]), sep='\t', file=output_file)
                    print(file=output_file)
                except Exception as e:
                    print("Error:", e, end='\n\n', file=output_file)

if __name__ == "__main__":
    main()
