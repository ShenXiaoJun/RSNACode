index=0
for file in /home/shenxj/RSNA/data/stage_1_train_jpg/*
do
	echo "$index"
	echo $file
	cp $file /home/shenxj/RSNA/code/test/pic_1k_train
	((index++))
	[ $index -eq 1000 ] && break		
done
