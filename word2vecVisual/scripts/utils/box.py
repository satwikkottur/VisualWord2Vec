# Wrapper around the shelf module
import shelve

def boxVariables(savePath, variables):
    my_shelf = shelve.open(savePath,'n') # 'n' for new

    for key in variables:
        try:
            my_shelf[key] = ()[key]
        except TypeError:
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()

def unboxVariables(loadPath):
    my_shelf = shelve.open(loadPath)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()

