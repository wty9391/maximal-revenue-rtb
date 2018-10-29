advertisers="1458"
root="/home/wty/sda2/spyder-workspace/make-ipinyou-data/"
for advertiser in $advertisers; do
    echo "run [python ../train_init.py $root/$advertiser/train.log.txt $root/$advertiser/test.log.txt $root/$advertiser/featindex.txt ../result/$advertiser]"
    mkdir -p ../result/$advertiser
    python ../train_init.py $root/$advertiser/train.log.txt $root/$advertiser/test.log.txt $root/$advertiser/featindex.txt ../result/$advertiser
done