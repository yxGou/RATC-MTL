function [X] = Unfold1( X, dim, i )
X = reshape(shiftdim(X,i-1), dim(i), []);