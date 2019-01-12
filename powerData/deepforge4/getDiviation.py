# -*- coding: utf-8 -*-


# Editing "ScoreModel" Implementation
# 
# The 'execute' method will be called when the operation is run



class GetDiviation():

    def execute(self, original, adv):
        assert(len(original) == len(adv))
        ret = []
        for idx in range(len(original)):
            rd = (adv[idx] - original[idx]) / original[idx]
            print('original: ' + str(original[idx]) + ', adv: ' + str(adv[idx]) + ', deviation: ' + str(rd))
            ret.append(rd)
            
        return ret
        

