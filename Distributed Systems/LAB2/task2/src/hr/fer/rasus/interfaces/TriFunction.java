package hr.fer.rasus.interfaces;

public interface TriFunction<T,V,R,S>{
	S apply(T t, V v, R r);
}
