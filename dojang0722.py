#30 함수 2
# def personal_info(name, age, address):
#     print('이름: ', name)
#     print('나이: ', age)
#     print('주소: ', address)
#
# #personal_info(name='홍길동', age=30, address='서울시 용산구 이촌동')
# x = {'name': '홍길동', 'age': 30, 'address': '서울시 용산구 이촌동'}
# personal_info(**x)  # 딕셔너리를 인자로  * 하나일 경우 키값 사용

#TODO 30 심사 문제
korean, english, mathematics, science = map(int, input().split())

def get_average(**kwargs):
    return float(sum(kwargs.values())/len(kwargs.keys()))
def get_min_max_score(*args):
    return min(args),max(args)

min_score, max_score = get_min_max_score(korean, english, mathematics, science)
average_score = get_average(korean=korean, english=english,
                            mathematics=mathematics, science=science)
print('낮은 점수: {0:.2f}, 높은 점수: {1:.2f}, 평균 점수: {2:.2f}'
      .format(min_score, max_score, average_score))

min_score, max_score = get_min_max_score(english, science)
average_score = get_average(english=english, science=science)
print('낮은 점수: {0:.2f}, 높은 점수: {1:.2f}, 평균 점수: {2:.2f}'
      .format(min_score, max_score, average_score))