require 'csvigo';
require 'image';

local data = {}


function data.getHash()
    local hash = {}
    for i=string.byte('0') ,string.byte('9')  do
        table.insert(hash,string.char(i))
    end
    for i=string.byte('A') ,string.byte('Z')  do
        table.insert(hash,string.char(i))
    end
    return hash
end


function data.getMap()
    local hash = data.getHash()
    local map = {}
    for k,v in ipairs(hash) do
        map[v] = k
    end
    return map
end


function data.loadY(dir)
    local csv = csvigo.load{path = dir .. 'ans.txt', mode = 'raw'}
    local Ystr = {}
    for i=1,#csv do
        table.insert(Ystr,string.upper(csv[i][1]))
    end
    local N = #Ystr
    local d = #Ystr[1]
    local Y = torch.zeros(N,d)
    local map = data.getMap()
    for i=1,N do
        for j=1,d do
            local c = string.sub(Ystr[i],j,j)
            Y[i][j] = map[c]
        end
    end
    return Y
end

function data.loadX(dir,pfx,N,dH,dW)
    local dH = dH or 50
    local dW = dW or 170
    local function readX(dir,i,b)
        local X = torch.zeros(b,dH,dW)
        for j=i,i+b-1 do
            X[j] = image.load(dir ..''..(pfx or '').. j ..'.png')[4]
        end
        return X
    end
    return readX(dir,1,N) 
end

function data.storeXY(dir,dH,dW,pfx)
    local Y = data.loadY(dir)
    local X = data.loadX(dir,pfx,Y:size(1),dH,dW)
    torch.save(dir .. '/data.t7',{X=X,Y=Y})
end


function data.loadXY(dir)
    local d = torch.load(dir .. '/data.t7')
    return d.X,d.Y
end


function data.split(X,Y,Nv)
    local N = X:size(1)
    local Nv = Nv or N/5
    local I = torch.randperm(N):long()
    local It = I[{{1,N-Nv}}]
    local Iv = I[{{N-Nv+1,N}}]
    local Xt,Yt = X:index(1,It),Y:index(1,It)
    local Xv,Yv = X:index(1,Iv),Y:index(1,Iv)
    return Xt,Yt,Xv,Yv
end

function data.convert1(Y1)
    local N = Y1:nElement()
    local Y2 = torch.ones(7)
    local j = 1
    for i=1,7 do
        if(Y1[i]~=1) then
            Y2[j] = Y1[i]
            j = j + 1
        end
    end
    return Y2
end

function data.convert(Y)
    for i=1,Y:size(1) do
        Y[i] = data.convert1(Y[i])
    end
end


return data
