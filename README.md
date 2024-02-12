# Построение таблицы истинности по для введённой булевой функции
Программа была написана в качестве лабораторной работы. Ввод и вывод осуществляется в файл, имена файлов 
передаются в параметрах программы.


Пример вызова:
```
python main.py <входной файл с расширением> <выходной файл с расширением>
```


Во входном файле булевые функции задаются в виде (можно строить таблицы истинности сразу для нескольких функций,
ошибки при обработке выводятся в выходной файл):
Пример:
```
(((!x = z) V (x ^ y)) & (x ^ !y)) > z
(x > y) > ((x > z) > (x > y & z))
!(((x ^ !y) V (!x > z)) & (y | z))
```

Пример результата:
```
ЛФ: (((!x=z)V(x^y))&(x^!y))>z
Найденные переменные: x y z
x	y	z	|	F
------------+-------
0	0	0	|	1
0	0	1	|	1
0	1	0	|	1
0	1	1	|	1
1	0	0	|	1
1	0	1	|	1
1	1	0	|	0
1	1	1	|	1

ЛФ: (x>y)>((x>z)>(x>y&z))
Найденные переменные: x y z
x	y	z	|	F
------------+-------
0	0	0	|	1
0	0	1	|	1
0	1	0	|	1
0	1	1	|	1
1	0	0	|	1
1	0	1	|	1
1	1	0	|	1
1	1	1	|	1

ЛФ: !(((x^!y)V(!x>z))&(y|z))
Найденные переменные: x y z
x	y	z	|	F
------------+-------
0	0	0	|	0
0	0	1	|	0
0	1	0	|	1
0	1	1	|	1
1	0	0	|	0
1	0	1	|	0
1	1	0	|	0
1	1	1	|	1
```

## Сведения о таблице истинности
Таблица истинности — это таблица, которая описывает логическую функцию и отражает все 
значения функции при всех возможных значениях её аргументов.


Таблицы истинности применяются в булевой алгебре и в цифровой электронной технике для описания работы логических схем.

## Описание работы программы
Есть возможность в словаре *char_to_keyword* поменять обозначения (символы) для базовых булевых операций.


Объект типа BoolVector представляет собой векторы-столбцы логических значений.


В основе работы программы лежит принцип математической индукции. Программа согласно приоритетам в цикле вычисляет 
значения подвыражений до тех пор, пока в итоге не получится объект типа BoolVector, а потом вычисленные значения
подвыражений использует для вычисления следующих выражений. В итоге получаем объект типа BoolVector, который и будет
олицетворять булевую функцию. В качестве изначальных переменных тоже выступают объекты типа BoolVector, значения
которых задаются программой.


Программа может обрабатывать следующие исключения:
- Число переданных программе параметров не равняется 2. 
Результат:
```
Неправильный вызов программы! Правильный вызов:
python Lab3.py <входной файл с расширением> <выходной файл с расширением>
```

- Была передана пустая строка во входном файле.
Результат: ```Пустая строка!```

- Была передана функция, содержащая переменную, несвязанную ни с каким оператором, или оператор, несвязанный ни с какой переменной. 
Результат: ```Неправильное задание ЛФ! (наличие лишней переменной или оператора)```

- Была передана функция, содержащая оператор, связанный не с переменной, а с другим операндом. 
Результат: ```Неправильное задание ЛФ! (Оператору в качестве операнда был передан оператор)```

- Была передана функция, содержащая лишнюю открывающую скобку. 
Результат: ```Неправильное задание ЛФ! (Не закрыта скобка)```

- Была передана функция, содержащая лишнюю закрывающую скобку. 
Результат: ```Неправильное задание ЛФ! (Лишние закрывающие скобки)```

