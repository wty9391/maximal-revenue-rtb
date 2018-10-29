advertisers="1458"
root="/home/wty/sda2/spyder-workspace/make-ipinyou-data/"
for advertiser in $advertisers; do
    echo "run [python ../train_bidder.py ../result/$advertiser $root/$advertiser/train.log.txt $root/$advertiser/test.log.txt]"
    mkdir -p ../result/$advertiser
    python ../train_bidder.py ../result/$advertiser $root/$advertiser/train.log.txt $root/$advertiser/test.log.txt
done