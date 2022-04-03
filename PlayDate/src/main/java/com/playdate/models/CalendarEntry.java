package com.playdate.models;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.playdate.enumerations.RepeatInterval;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import javax.persistence.*;
import javax.validation.constraints.NotEmpty;
import javax.validation.constraints.NotNull;
import java.util.Date;
import java.util.List;

@Entity
@Table(name = "calendar_entries")
@Data
@EqualsAndHashCode(of = "id")
public class CalendarEntry{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotNull
    @Temporal(TemporalType.TIMESTAMP)
    private Date start;

    @NotNull
    @Temporal(TemporalType.TIMESTAMP)
    private Date end;

    @Enumerated(EnumType.STRING)
    private RepeatInterval repeatInterval;

    @Temporal(TemporalType.TIMESTAMP)
    private Date pauseFrom;

    @Temporal(TemporalType.TIMESTAMP)
    private Date pauseTo;

    @NotEmpty(message = "Event should have title")
    private String title;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "calendar_id")
    @ToString.Exclude
    @JsonIgnore
    private Calendar calendar;
}
