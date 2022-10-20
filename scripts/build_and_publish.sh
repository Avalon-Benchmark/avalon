set -e

THIS_FILE=$(realpath "$0")
THIS_PATH=$(dirname "${THIS_FILE}")
AVALON_DIR=$(realpath $THIS_PATH/..)

cd $AVALON_DIR
rm -rf dist/

pip install --quiet --upgrade twine build
python -m build

read -p "Where do you want to publish (pypi | testpypi | cancel)? " repo
if [[ $repo != pypi && $repo != testpypi ]]
then
  echo "not publishing to $repo"
  exit
fi

python -m twine upload --repository $repo dist/*