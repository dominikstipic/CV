package hr.fer.zemris.java.hw05.db;

/**
 * Omugucava dohvacanje atributa danog studenta u vrijeme izvodenja programa.
 * Atributi koje student sadrzi su : jmbag,ime,prezime
 * @author Dominik Stipic
 *
 */
public interface IFieldValueGetter {
	/**
	 * dohvaca studentov atribut koji je naveden u upitu 
	 * @param record student ciji se atribut provjerava
	 * @return atribut
	 */
	String get(StudentRecord record);
}
