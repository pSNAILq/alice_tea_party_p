def kirei(text = "ももたろう は 。 おじいさん を 「 あい 「 」 ました 」  。 しかし 、 "):
    n = text.rfind('。')
    text2 = text[1:n+1]
    result = ''.join(text2)
    # print(result)
    return result
    # for word in t:
    #     if '「' == word:
    #         if ch == 1:
    #             del t[count]
    #         ch = 1
    #     elif '」' == word:
    #         if ch == 0:
    #             del t[count]
    #         ch = 0
    #     count += 1

    # if 1 == ch:
    #     #text += '」'
    #     text.append('」') 

    # print(t)