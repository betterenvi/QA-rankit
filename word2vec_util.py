def readEmbedFile(embedFile):  
    input = open(embedFile,'r')  
    lines= []  
    for line in input:  
        lines.append(line)  
    nwords = len(lines)
    splits = lines[0].split(' ')
    dim = len(splits) - 1
    embeddings = dict()
    for s in lines:  
        splits = s.split(' ')  
        embeddings[splits[0]] = map(float, splits[1:dim+1])  
    return embeddings  



