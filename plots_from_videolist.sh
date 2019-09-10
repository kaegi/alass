#!/usr/bin/bash
PREFIX=./generated-data

export VIDEOLIST_FILE="./videos.list"
export CACHE_DIR="$PREFIX/cache"
export DATABASE_DIR="$PREFIX/1-database"
export STATISTICS_DIR="$PREFIX/2-statistics"
export PLOTS_DIR="$PREFIX/3-plots"


skip_database=1
skip_statistics=0
skip_plots=0
clean_cache=0

while [ "$1" != "" ]; do
    case $1 in
        --no-skip-database)    skip_database=0 ;;
        --skip-database)    skip_database=1 ;;
        --skip-statistics)    skip_statistics=1 ;;
        --skip-plots)    skip_plots=1 ;;
        --clean-cache)    clean_cache=1 ;;
        * )                     echo "Unexpected input flag '$1'! Exiting."
                                exit 1
    esac
    shift
done

if [ "$clean_cache" = "1" ]; then
	echo "Cleaning statistics cache requested otherwise by user!"
	rm "$CACHE_DIR" -rf
fi



echo '======================================================================'
echo "Generating database..."
echo '======================================================================'


if [ "$skip_database" = "1" ]; then
	echo "Skipping database creation as not requested otherwise by user!"
else
	python3.7 ./statistics-helpers/generate_database_from_videolist.py --videolist-file "$VIDEOLIST_FILE" --database-dir "$DATABASE_DIR"
	echo "Generating database done!"
fi

echo
echo

echo '======================================================================'
echo "Generating statistics..."
echo '======================================================================'

if [ "$skip_statistics" = "1" ]; then
	echo "Skipping statistics generation as requested by user!"
else
	cargo run \
		--release \
		--example=generate_statistics_from_database \
		-- \
		--database-dir "$DATABASE_DIR" --statistics-dir "$STATISTICS_DIR" --cache-dir "$CACHE_DIR" \
		--split-penalties 0.05,0.1,0.25,0.5,1,2,3,4,5,6,7,8,9,10,20,30,50,100,1000 \
		--default-split-penalty 6 \
		--default-min-span-length 200 \
		--default-optimization 2 \
		--default-max-good-sync-offset 300,500,1000,1500 \
		--default-required-good-sync-spans-percentage 25,50,75,95  \
		--num-threads 4 \
		#--split-penalties 0.25,0.5,1,2,3,4,5,6,8,10,30,100 \
		#--quiet
		#--clean-cache-line-pairs \
 # 200ms/25%, 500ms/50%, 1000ms/75%, 1500ms/95%
	echo "Generating statistics done!"
fi

echo
echo

echo '======================================================================'
echo "Generating plots in '${PLOTS_DIR}'..."
echo '======================================================================'
if [ "$skip_plots" = "1" ]; then
	echo "Skipping plots generation as requested by user!"
else
	python3.7 ./statistics-helpers/generate_plots_from_statistics.py --statistics-dir "$STATISTICS_DIR" --plots-dir "$PLOTS_DIR"
	echo "Generating plots in '${PLOTS_DIR}' done!"
fi

