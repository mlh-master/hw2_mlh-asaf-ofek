

def str_to_bool_seriess(s):
    for index, value in s.items():
        if(value=='Yes'):
            s[index] = True
        elif(value=='No'):
            s[index] = False
        elif(value=='Male'):
            s[index] = True
        elif(value == 'Female'):
            s[index] = False
        elif(value=='Positive'):
            s[index] = True
        elif (value == 'Negative'):
            s[index] = False
#         elif(value==1):
#             s[index] = True
#         elif(value==0):
#             s[index] = False
    return s