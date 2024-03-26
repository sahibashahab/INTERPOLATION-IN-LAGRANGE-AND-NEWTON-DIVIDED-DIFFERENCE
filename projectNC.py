'''
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk

def lagrange_interpolation(x, y, x_interp):
    n = len(x)
    y_interp = []

    for xi in x_interp:
        yi = 0.0
        for j in range(n):
            term = 1.0
            for k in range(n):
                if k != j:
                    term *= (xi - x[k]) / (x[j] - x[k])
            yi += y[j] * term
        y_interp.append(yi)

    return y_interp

def divided_difference(x, y):
    n = len(x)
    coefficients = np.zeros((n, n))
    coefficients[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coefficients[i][j] = (coefficients[i+1][j-1] - coefficients[i][j-1]) / (x[i+j] - x[i])

    return coefficients[0]


def newton_interpolation(x, y, x_interp):
    coefficients = divided_difference(x, y)
    n = len(x)
    y_interp = []

    for xi in x_interp:
        yi = coefficients[n - 1]
        for k in range(n - 2, -1, -1):
            yi = coefficients[k] + (xi - x[k]) * yi
        y_interp.append(yi)

    return y_interp


def calculate_interpolation():
    x_pts = [float(x) for x in entry_x_pts.get().split(',')]
    y_pts = [float(y) for y in entry_y_pts.get().split(',')]
    x_vals = [float(x) for x in entry_x_vals.get().split(',')]

    if selected_method.get() == 'Lagrange':
        y_interp = lagrange_interpolation(x_pts, y_pts, x_vals)
    else:
        y_interp = newton_interpolation(x_pts, y_pts, x_vals)

    messagebox.showinfo("Interpolation Result", "Interpolated Values:\n{}".format(y_interp))

    plot_graph(x_pts, y_pts, x_vals, y_interp)





def plot_graph(x_pts, y_pts, x_interp, y_interp):
    plt.figure(figsize=(8, 6))
    plt.plot(x_pts, y_pts, 'ro', label='Data Points')

    # Generate a smooth line for interpolation
    x_smooth = np.linspace(min(x_pts), max(x_pts), 100)
    if selected_method.get() == 'Lagrange':
        y_smooth = lagrange_interpolation(x_pts, y_pts, x_smooth)
    else:
        y_smooth = newton_interpolation(x_pts, y_pts, x_smooth)
    plt.plot(x_smooth, y_smooth, 'b-', label='Interpolation')

    plt.scatter(x_pts, y_pts, color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Interpolation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

window = tk.Tk()
window.title("Interpolation")
window.geometry("400x300")

label_x_pts = tk.Label(window, text="Enter x data points:")
label_x_pts.pack()

entry_x_pts = tk.Entry(window)
entry_x_pts.pack()

label_y_pts = tk.Label(window, text="Enter y data points:")
label_y_pts.pack()

entry_y_pts = tk.Entry(window)
entry_y_pts.pack()

label_x_vals = tk.Label(window, text="Enter x values to interpolate:")
label_x_vals.pack()
entry_x_vals = tk.Entry(window)
entry_x_vals.pack()
label_method = tk.Label(window, text="Select interpolation method:")
label_method.pack()

selected_method = tk.StringVar()
selected_method.set("Lagrange")

radio_lagrange = tk.Radiobutton(window, text="Lagrange", variable=selected_method, value="Lagrange")
radio_lagrange.pack()

radio_newton = tk.Radiobutton(window, text="Newton Divided Difference", variable=selected_method, value="Newton")
radio_newton.pack()

button_calculate = tk.Button(window, text="Calculate", command=calculate_interpolation)
button_calculate.pack()

window.mainloop()
'''
# Conversion from Regex to NFA
import json
import sys

non_symbols = ['+', '*', '.', '(', ')']
nfa = {}

class charType:
    SYMBOL = 1
    CONCAT = 2
    UNION  = 3
    KLEENE = 4

class NFAState:
    def _init_(self):
        self.next_state = {}

class ExpressionTree:
    def _init_(self, charType, value=None):
        self.charType = charType
        self.value = value
        self.left = None
        self.right = None    

def make_exp_tree(regexp):
    stack = []
    for c in regexp:
        if c == "+":
            z = ExpressionTree(charType.UNION)
            z.right = stack.pop()
            z.left = stack.pop()
            stack.append(z)
        elif c == ".":
            z = ExpressionTree(charType.CONCAT)
            z.right = stack.pop()
            z.left = stack.pop()
            stack.append(z)
        elif c == "*":
            z = ExpressionTree(charType.KLEENE)
            z.left = stack.pop() 
            stack.append(z)
        elif c == "(" or c == ")":
            continue  
        else:
            stack.append(ExpressionTree(charType.SYMBOL, c))
    return stack[0]


def compPrecedence(a, b):
    p = ["+", ".", "*"]
    return p.index(a) > p.index(b)


def compute_regex(exp_t):
    # returns E-NFA
    if exp_t.charType == charType.CONCAT:
        return do_concat(exp_t)
    elif exp_t.charType == charType.UNION:
        return do_union(exp_t)
    elif exp_t.charType == charType.KLEENE:
        return do_kleene_star(exp_t)
    else:
        return eval_symbol(exp_t)


def eval_symbol(exp_t):
    start = NFAState()
    end = NFAState()
    
    start.next_state[exp_t.value] = [end]
    return start, end


def do_concat(exp_t):
    left_nfa  = compute_regex(exp_t.left)
    right_nfa = compute_regex(exp_t.right)

    left_nfa[1].next_state['$'] = [right_nfa[0]]
    return left_nfa[0], right_nfa[1]


def do_union(exp_t):
    start = NFAState()
    end = NFAState()

    first_nfa = compute_regex(exp_t.left)
    second_nfa = compute_regex(exp_t.right)

    start.next_state['$'] = [first_nfa[0], second_nfa[0]]
    first_nfa[1].next_state['$'] = [end]
    second_nfa[1].next_state['$'] = [end]

    return start, end


def do_kleene_star(exp_t):
    start = NFAState()
    end = NFAState()

    starred_nfa = compute_regex(exp_t.left)

    start.next_state['$'] = [starred_nfa[0], end]
    starred_nfa[1].next_state['$'] = [starred_nfa[0], end]

    return start, end


def arrange_transitions(state, states_done, symbol_table):
    global nfa

    if state in states_done:
        return

    states_done.append(state)

    for symbol in list(state.next_state):
        if symbol not in nfa['letters']:
            nfa['letters'].append(symbol)
        for ns in state.next_state[symbol]:
            if ns not in symbol_table:
                symbol_table[ns] = sorted(symbol_table.values())[-1] + 1
                q_state = "Q" + str(symbol_table[ns])
                nfa['states'].append(q_state)
            nfa['transition_function'].append(["Q" + str(symbol_table[state]), symbol, "Q" + str(symbol_table[ns])])

        for ns in state.next_state[symbol]:
            arrange_transitions(ns, states_done, symbol_table)

def notation_to_num(str):
    return int(str[1:])

def final_st_dfs():
    global nfa
    for st in nfa["states"]:
        count = 0
        for val in nfa['transition_function']:
            if val[0] == st and val[2] != st:
                count += 1
        if count == 0 and st not in nfa["final_states"]:
            nfa["final_states"].append(st)


def arrange_nfa(fa):
    global nfa
    nfa['states'] = []
    nfa['letters'] = []
    nfa['transition_function'] = []
    nfa['start_states'] = []
    nfa['final_states'] = []
    q_1 = "Q" + str(1)
    nfa['states'].append(q_1)
    arrange_transitions(fa[0], [], {fa[0] : 1})
    
    st_num = [notation_to_num(i) for i in nfa['states']]

    nfa["start_states"].append("Q1")
    # nfa["final_states"].append("Q" + str(sorted(st_num)[-1]))
    # final_st_dfs(nfa["final_states"][0])
    final_st_dfs()


def add_concat(regex):
    global non_symbols
    l = len(regex)
    res = []
    for i in range(l - 1):
        res.append(regex[i])
        if regex[i] not in non_symbols:
            if regex[i + 1] not in non_symbols or regex[i + 1] == '(':
                res += '.'
        if regex[i] == ')' and regex[i + 1] == '(':
            res += '.'
        if regex[i] == '*' and regex[i + 1] == '(':
            res += '.'
        if regex[i] == '*' and regex[i + 1] not in non_symbols:
            res += '.'
        if regex[i] == ')' and regex[i + 1] not in non_symbols:
            res += '.'

    res += regex[l - 1]
    return res


def compute_postfix(regexp):
    stk = []
    res = ""

    for c in regexp:
        if c not in non_symbols or c == "*":
            res += c
        elif c == ")":
            while len(stk) > 0 and stk[-1] != "(":
                res += stk.pop()
            stk.pop()
        elif c == "(":
            stk.append(c)
        elif len(stk) == 0 or stk[-1] == "(" or compPrecedence(c, stk[-1]):
            stk.append(c)
        else:
            while len(stk) > 0 and stk[-1] != "(" and not compPrecedence(c, stk[-1]):
                res += stk.pop()
            stk.append(c)

    while len(stk) > 0:
        res += stk.pop()

    return res

def polish_regex(regex):
    reg = add_concat(regex)
    regg = compute_postfix(reg)
    return regg


def load_regex():
    with open('q1_eg_in_regex.json', 'r') as inpjson:
        regex = json.loads(inpjson.read())
    return regex

def output_nfa():
    global nfa
    with open('q1_eg_out_NFA.json', 'w') as outjson:
        outjson.write(json.dumps(nfa, indent = 4))
        
if _name_ == "_main_":
    r = load_regex()
    reg = r['regex']
    pr = polish_regex(reg)
    et = make_exp_tree(pr)
    fa = compute_regex(et)
    arrange_nfa(fa)
    output_nfa()