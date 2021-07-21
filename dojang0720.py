# 27. 파일 사용하기
# file = open('hello.txt', 'w')    # hello.txt 파일을 쓰기 모드(w)로 열기. 파일 객체 반환
# file.write('Hello, world!')      # 파일에 문자열 저장
# file.close()                     # 파일 닫기
#
# file = open('hello.txt', 'r')    # hello.txt 파일을 읽기 모드(r)로 열기. 파일 객체 반환
# s = file.read()                  # 파일에서 문자열 읽기
# print(s)                         # Hello, world!
# file.close()                     # 파일 객체 닫기
#
# with open('hello.txt', 'r') as file:    # hello.txt 파일을 읽기 모드(r)로 열기
#     s = file.read()                     # 파일에서 문자열 읽기
#     print(s)                            # Hello, world!  // 자동으로 파일 객체 닫기

# # TODO 27 심사문제
# with open("C:/Users/82105/PycharmProjects/mathstudy/words.txt",'r') as file:
#     words = file.read()
#     wr = words.split(' ')
#     for word in wr:
#         if 'c' in word:
#             print(word.strip(',.'))

# 28.회문 판별하기
# word = input('단어를 입력하세요: ')
#
# is_palindrome = True  # 회문 판별값을 저장할 변수, 초깃값은 True
# for i in range(len(word) // 2):  # 0부터 문자열 길이의 절반만큼 반복
#     if word[i] != word[-1 - i]:  # 왼쪽 문자와 오른쪽 문자를 비교하여 문자가 다르면
#         is_palindrome = False  # 회문이 아님
#         break
#
# print(is_palindrome)  # 회문 판별값 출력
# TODO 28 심사문제
# with open("C:/Users/82105/PycharmProjects/mathstudy/words.txt",'r') as file:
#     word = file.readlines()
# for words in word:
#     words = words.strip('\n')
#     if words == words[::-1]:
#         print(words)
#
# read()
#
# 파일에서 문자열을 읽음
#
# write('문자열')
#
# 파일에 문자열을 씀
#
# readline()
#
# 파일의 내용을 한 줄 읽음
#
# readlines()
#
# 파일의 내용을 한 줄씩 리스트 형태로 가져옴
#
# writelines(문자열리스트)
#
# 파일에 리스트의 문자열을 씀, 리스트의 각 문자열에는 \n을 붙여주어야 함
#
# pickle.load(파일객체)
#
# 파일에서 파이썬 객체를 읽음
#
# pickle.dump(객체, 파일객체)
#
# 파이썬 객체를 파일에 저장