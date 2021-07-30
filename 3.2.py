import numpylearn as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.integrate import quad
import scipy
x= sp.Symbol('x')
print(sp.diff(3*x**2+1,x))

#적분
a= sp.Symbol('a')
print(sp.integrate(3.0*a**2+1,a))

#3x^2+1 미분하기
print('예제1--------------------------------------------------------')
b = sp.Symbol('b')
print(sp.diff(3.0*b**2+1,b))

print('예제2--------------------------------------------------------')
line = lambda c: c+3
solution = fsolve(line,-2)
print(solution)

print('예제3--------------------------------------------------------')
#적분할 함수 정의
func = lambda d : np.cos(np.exp(d))**2
#0부터 3까지 구간 함수 적분
sol = quad(func,0,3)
print(sol)