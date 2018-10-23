index=0
for file in /home/shenxj/RSNA/data/stage_1_test_jpg/*
do
	echo "$index"
	echo $file
	cp $file /home/shenxj/RSNA/code/test/pic_100_test
	((index++))
	[ $index -eq 100 ] && break		
done
