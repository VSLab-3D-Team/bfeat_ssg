from collections import OrderedDict
import random, time

class Replay_Buffer:
    ## SET      : 값의 중복 미허용  / 키당 집합으로 복수의 값 저장  / 
    ## LIST     : 값의 중복 허용    / 키당 리스트로 복수의 값 저장  /
    ## SINGLE   : 값의 중복 허용    / 키당 하나의 값만 저장         /
    ## replace시 비복원 추출
    ## 중복여부와 상관없이 입력으로 key 값이 들어오면 최근사용한것으로 간주
    ## 사이즈가 꽉차면 가장 오래 안쓰인 키의 값들 부터 삭제 LIST나 SET의 경우 해당컨테이너의 모든요소 일괄 삭제
    def __init__(self, buffer_size : int, type="SET", replace=True):
        self.buffer_size=buffer_size
        self.type=type
        self.replace=replace
        self.cache = OrderedDict()
        self._size=0
        assert type in ["SET", "LIST", "SINGLE"]
        
    def __str__(self):
        tempstr=""#f"size : {self._size}/{self.buffer_size}, type : {self.type}, IsReplace : {self.replace}\n"
        tempstr+=', '.join(f"{k}: {v}" for k, v in self.cache.items())
        tempstr+="\n"
        
        return tempstr
    def _Convert_triplet_to_key(self,triplet):
        key=triplet
        return key
    
    def _Convert_key_to_triplet(self,key):
        triplet=key
        return triplet
    
    def _Add_sample(self, gt_key : int, wrong_key : int):
        if self.type=="SET":
            if gt_key not in self.cache:
                self.cache[gt_key]=set()
            if wrong_key not in self.cache[gt_key]:
                self.cache[gt_key].add(wrong_key)
                self._size+=1
            
        elif self.type=="LIST":
            self.cache.setdefault(gt_key, []).append(wrong_key)
            self._size+=1
            
        elif self.type=="SINGLE":
            if gt_key not in self.cache:
                self._size+=1
            self.cache[gt_key] = wrong_key
            
        else:
            return True
        
        self.cache.move_to_end(gt_key) 
        return False
    
    def _Remove_sample(self):
        item=self.cache.popitem(last=False)[1]
        if self.type=="SET":
            self._size-=len(item)
            
        elif self.type=="LIST":
            self._size-=len(item)
            
        elif self.type=="SINGLE":
            self._size-=1
            
        else:
            return True
        
        return False
    
    def Put_Sample(self,gt_triplet,wrong_triplet):
        gt_key=self._Convert_triplet_to_key(gt_triplet)
        wrong_key=self._Convert_triplet_to_key(wrong_triplet)
        
        self._Add_sample(gt_key,wrong_key)
        
        if self._size > self.buffer_size:
            self._Remove_sample()
    
    def Get_Sample(self,anchor_triplet):
        anchor_key=self._Convert_triplet_to_key(anchor_triplet)
        if anchor_key not in self.cache:
            return None
        
        if self.type=="SET":
            return_key=random.choice(list(self.cache[anchor_key]))
            self._size-=1
            if self.replace: 
                self.cache[anchor_key].remove(return_key)
            #else:
            #    self.cache.move_to_end(anchor_key)
            if not self.cache[anchor_key]:
                del self.cache[anchor_key]
                
        elif self.type=="LIST":
            return_key=random.choice(self.cache[anchor_key])
            self._size-=1
            if self.replace: 
                self.cache[anchor_key].remove(return_key)
            #else:
            #    self.cache.move_to_end(anchor_key)
            if not self.cache[anchor_key]:
                del self.cache[anchor_key]
            
        elif self.type=="SINGLE":
            return_key=self.cache[anchor_key]
            if self.replace: 
                self.cache.pop(anchor_key)
                self._size-=1
            #else:
            #    self.cache.move_to_end(anchor_key)
            
        return self._Convert_key_to_triplet(return_key)


##버퍼의 실제 작동 체크 / 실제 값 잘나오는지 확인용
def test1():
    SinglevBuffer=Replay_Buffer(5,"SINGLE",True)
    SetvBuffer=Replay_Buffer(5,"SET",True)
    ListvBuffer=Replay_Buffer(5,"LIST",True)
    Buffer_list=[SinglevBuffer,SetvBuffer,ListvBuffer]
    type_index=2
    Buffer=Buffer_list[type_index]
    vlist=[(1,2,0),(1,3,0),(2,4,0),(3,5,1),(4,6,0),(5,7,0),(3,9,1),(1,2,0),(3,5,0),(4,6,1),(5,7,0),(3,9,0),(1,2,1),(3,9,1),(1,2,0)]
    print("\n\ntest code\n\n")
    for key, value, type in vlist:
        if type==0:
            print(f"add ({key} , {value})")
            Buffer.Put_Sample(key,value)
        else:
            print(f"get {key}")
            if ((temp:=Buffer.Get_Sample(key))==None):
                print("no sample")
            else:
                print(f"get sample : {temp}")
        print(Buffer)

## 버퍼의 시간복잡도 체크용/ 버퍼사이즈/컨테이너 별 테스트 
## BUFFER_SIZE  |SINGLE     |LIST       |SET        | 
## 1E8          |0.25       |0.56       |0.60       | 
## 1E6          |0.28       |0.35       |0.40       |
## 1E4          |0.22       |0.25       |0.25       |
## 한 EPOCH당 예상 소요시간 / 500 epoch 까지만 / replace : True / 
## 전체사이즈가 커지면서 생기는 차이는 적음             => 삽입/삭제/검색 O(1)에서 동작 / SINGLE의 경우 전체사이즈가 665600으로 제한되어 더 시간은 안\는다.
##
## 컨테이너별 차이가 있긴하나 크지는 않다               => 확실히 키당 값이 하나인 싱글이 빠르고 나머지는 비슷하나 생각보다 적은 차이
## 전체 버퍼사이즈가 커질수록 약간 느려지기함           =>  하지만 그 차이가 100배 차이시 2배 차이가 나는 듯 큰 차이는 아님
def test2():
    buffer_size=1E4
    triplet_num=160*160*26 # triplet 조합의 수
    triplet_per_epoch=int(2E5) #한 epoch당 예상 triplet의수
    epoch=800
    Buffer=Replay_Buffer(buffer_size,"SET",True)
    print("\n\ntest code\n\n")
    print("pp")
    key_list=   [random.randint(0, triplet_num) for _ in range(triplet_per_epoch)]
    value_list= [random.randint(0, triplet_num) for _ in range(triplet_per_epoch)]
    type_list=  [random.randint(0, 2) for _ in range(triplet_per_epoch)]
    print("start_test")
    srt_time=time.time()
    for step in range(epoch):
        for i in range(triplet_per_epoch):
            if type_list[i]!=2:
                Buffer.Put_Sample((key_list[i]+step)%triplet_num,(value_list[i]+step)%triplet_num)
            else:
                Buffer.Get_Sample((key_list[i]+step)%triplet_num)
        end_time=time.time()
        elapsed_time=end_time-srt_time
        srt_time=end_time
        print(f"{step} : {elapsed_time}")
        
## 버퍼의 공간복잡도 체크용/ 버퍼사이즈/컨테이너 별 테스트 
## BUFFER_SIZE  |SINGLE     |LIST       |SET        | 
## 1E8          |93.2M      |4.17G      |8.8G       | 
## 1E6          |93.2M      |134M       |203M       |
## 1E4          |1.81M      |2.67M      |3.93M      |
## empty        |856        |856        |856        |
## SINGLE의경우 665600으로 크기가 제한되어 버퍼사이즈와 상관없이 결국 수렴
## LIST와 SET의 메모리는 2배정도 차이남
## 현재 key, value는 int 기준(28B) 문자열을 키로 삼으면 더 커질 수 있음 (대략 2배 정도=49+10*3) => 실제 order_dict
def test3():
    from pympler import asizeof
    
    buffer_size=1E6
    triplet_num=160*160*26 # triplet 조합의 수
    triplet_per_epoch=int(2E5) #한 epoch당 예상 triplet의수
    epoch=1000
    Buffer=Replay_Buffer(buffer_size,"LIST",True)
    print("\n\ntest code\n\n")
    print("pp")
    key_list=   [random.randint(0, triplet_num) for _ in range(triplet_per_epoch)]
    value_list= [random.randint(0, triplet_num) for _ in range(triplet_per_epoch)]
    type_list=  [random.randint(0, 2) for _ in range(triplet_per_epoch)]
    print("start_test")
    print(f"empty {Buffer.type} size : { asizeof.asizeof(Buffer)}")
    for step in range(epoch):
        for i in range(triplet_per_epoch):
            if type_list[i]!=2:
                Buffer.Put_Sample((key_list[i]+step)%triplet_num,(value_list[i]+step)%triplet_num)
            else:
                continue
                Buffer.Get_Sample((key_list[i]+step)%triplet_num)
        print(f"{step} : {Buffer._size}")
        if Buffer._size>Buffer.buffer_size-1000:
            break;
    print(f"{buffer_size} {Buffer.type} size : { asizeof.asizeof(Buffer)}")
        

if __name__=="__main__":
    #test1()
    #test2()
    test3()