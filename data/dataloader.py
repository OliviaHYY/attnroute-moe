%%bash
# 1. Clean up any broken previous extractions
rm -rf /content/tiny-imagenet-200
rm -rf /content/tiny-imagenet-200/val_organized

# 2. Download the source dataset if it isn't there
if [ ! -f "/content/tiny-imagenet-200.zip" ]; then
    echo "Downloading Tiny ImageNet..."
    wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O /content/tiny-imagenet-200.zip
fi

# 3. Unzip the file cleanly
echo "Unzipping data structure..."
unzip -q /content/tiny-imagenet-200.zip -d /content/

# 4. Create the required validation folder structure manually
echo "Organizing validation folder split..."
cd /content/tiny-imagenet-200
mkdir -p val_organized

# Read the annotations file and copy images into class-named folders
while read -r img cls x1 y1 x2 y2; do
    mkdir -p "val_organized/$cls"
    cp "val/images/$img" "val_organized/$cls/$img"
done < val/val_annotations.txt

echo "✓ Data environment setup complete! 'val_organized' is ready."
