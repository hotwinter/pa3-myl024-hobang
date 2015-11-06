valgrind --tool=cachegrind ./apf -n 256 -i 2000
g_annotate --auto=yes cachegrind.out.1527  > Report.txt
