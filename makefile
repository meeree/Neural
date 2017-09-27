GDB = 
BLAS=-lblas 
LAPACK=-llapack 
ARMA=-I /home/jhazelden/Downloads/armadillo-7.500.2/include -DARMA_DONT_USE_WRAPPER
CFLAGS = -std=c++11 $(GDB) 


neural : neural.o
	g++ neural.o -o neural $(CFLAGS) $(BLAS) $(LAPACK)

neural.o : neural.cpp
	g++ -c neural.cpp $(CFLAGS) $(ARMA)

clean : 
	rm *.o neural
