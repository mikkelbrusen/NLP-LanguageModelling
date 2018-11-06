mkdir -p save
mkdir -p data
cd data

echo "- Downloading Penn Treebank (PTB)"
curl -s -o simple-examples.tgz http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p penn
cd penn
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.test.txt test.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt
cd ..

rm -rf simple-examples/

echo "---"
echo "Done setting up data"