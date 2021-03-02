package hr.fer.zemris.java.hw05.db;

/**
 * Modelira genericki Filter koji zna filtrirati studente s nekim
 * odredenim pravilom.
 * @author Dominik Stipic
 *
 */
public interface IFilter {
	/**
	 * Provjerava dali student zadovoljava neku pravilo pomocu kojeg se vrsi 
	 * filtriranje
	 * @param record student kojeg se filtrira
	 * @return true - student ce biti filtriran, false - student ce biti odbacen
	 */
	boolean accepts(StudentRecord record);
}
