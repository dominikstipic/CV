package hr.fer.zemris.java.hw05.db;

/**
 * Razred koji sadrzi metode koje dohvacaju atibute od studenta:
 * ime,prezime,jmbag
 * @author Dominik Stipic
 *
 */
public class FieldValueGetters {
	/**
	 * Dohvaca ime
	 */
	public static final IFieldValueGetter FIRST_NAME = (r -> r.getFirstName());
	/**
	 *Dohvaca prezime 
	 */
	public static final IFieldValueGetter LAST_NAME = (r -> r.getLastName());
	/**
	 * Dohvaca jmbag
	 */
	public static final IFieldValueGetter JMBAG = (r -> r.getJMBAG());
	
}
