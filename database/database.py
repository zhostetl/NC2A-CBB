from sqlalchemy import create_engine, Column, Integer, Float, String, Sequence, Date, Time, DateTime, ForeignKey, Boolean, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, aliased


# Setup SQLAlchemy
engine = create_engine('sqlite:///ncaa_basketball.db')
Base = declarative_base()

# Define the Games model
class Games(Base):
    __tablename__ = 'games'
    id = Column(Integer, Sequence('game_id_seq'), primary_key=True)
    
    date = Column(Date)
    game_id = Column(Integer())
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    season_id = Column(Integer, ForeignKey('season.id'), nullable=False)
    conference_id = Column(Integer, ForeignKey('conferences.id'), nullable=False)

    game_num = Column(Integer())
    game_location = Column(String(50))
    game_state = Column(String(50))

    opponent_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    opponent_conference_id = Column(Integer, ForeignKey('conferences.id'), nullable=True)
    
    two_point_field_goals_made = Column(Integer())
    two_point_field_goals_attempted = Column(Integer())
    two_point_field_goal_percentage = Column(Float(10))

    three_point_field_goals_made = Column(Integer())
    three_point_field_goals_attempted = Column(Integer())
    three_point_field_goal_percentage = Column(Float(10))

    field_goals_made = Column(Integer())
    field_goals_attemped = Column(Integer())
    field_goal_percentage = Column(Float(10))

    free_throws_made = Column(Integer())
    free_throws_attempted = Column(Integer())
    free_throw_percentage = Column(Float(10))

    rebounds = Column(Integer())
    offensive_rebounds = Column(Integer())
    defensive_rebounds = Column(Integer())

    assists = Column(Integer())
    steals = Column(Integer())
    blocks = Column(Integer())

    total_turnovers = Column(Integer())
    points_off_turnovers = Column(Integer())
    fast_break_points = Column(Integer())
    points_in_paint = Column(Integer())

    fouls = Column(Integer())
    technical_fouls = Column(Integer())
    flagrant_fouls = Column(Integer())

    largest_lead = Column(Integer())
    points = Column(Integer())
    win =  Column(Integer())
    loss = Column(Integer())
    home = Column(Integer())
    away = Column(Integer())
    neutral = Column(Integer())

    #four factors + efficiency stats
    eFG = Column(Float(10))
    TO_rate = Column(Float(10))
    OREB_per = Column(Float(10))
    FT_rate = Column(Float(10))

    opp_eFG = Column(Float(10))
    opp_TO_rate = Column(Float(10))
    DREB_per = Column(Float(10))
    opp_FT_rate = Column(Float(10))

    possessions = Column(Float(10))
    offensive_efficiency = Column(Float(10))
    defensive_efficiency = Column(Float(10))
    pace = Column(Float(10))
    
    distance_traveled = Column(Float(10), nullable=False)
    # distance_traveled = relationship('GameLocations', backref='games')

    attendance = Column(Integer())
    referee1 = Column(String(70))
    referee2 = Column(String(70))
    referee3 = Column(String(70))

    betting_line = Column(String(50))
    over_under = Column(String(10))


class Teams(Base):
    __tablename__ = 'teams'
    id = Column(Integer, Sequence('team_id_seq'), primary_key=True)
    name = Column(String(70), unique=True, nullable=False)
    espn_name = Column(String(70), unique=True, nullable=False)
    location = Column(String(70), nullable=False)

class Conferences(Base):
    __tablename__ = 'conferences'
    id = Column(Integer, Sequence('conference_id_seq'), primary_key=True)
    name = Column(String(70), unique=True, nullable=False)
    espn_group = Column(Integer(), unique=True, nullable=False)

class Season(Base):
    __tablename__ = 'season'
    id = Column(Integer, Sequence('season_id_seq'), primary_key=True)
    year = Column(Integer(), unique=True, nullable=False)
    
class TeamSeasonConference(Base):
    __tablename__ = 'team_season_conference'
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    season_id = Column(Integer, ForeignKey('season.id'), nullable=False)
    conference_id = Column(Integer, ForeignKey('conferences.id'), nullable=False)

class GameID(Base):
    __tablename__ = 'game_id'
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, nullable=False)
    season_id = Column(Integer, ForeignKey('season.id'), nullable=False)

class GameLocations(Base):
    __tablename__ = 'game_locations'
    id = Column(Integer, primary_key=True)
    location = Column(String(50), nullable=False)
    team_location = Column(String(50), nullable=False)
    distance = Column(Float(10), nullable=False)

class AdjustedMetrics(Base):
    __tablename__ = 'adjusted_metrics'
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    adj_efg_percentage = Column(Float(10), nullable=False)
    adj_turnover_percentage = Column(Float(10), nullable=False)
    adj_offensive_rebound_percentage = Column(Float(10), nullable=False)
    adj_free_throw_rate = Column(Float(10), nullable=False)
    opp_adj_efg_percentage = Column(Float(10), nullable=False)
    opp_adj_turnover_percentage = Column(Float(10), nullable=False)
    adj_def_rebound_percentage = Column(Float(10), nullable=False)
    opp_adj_free_throw_rate = Column(Float(10), nullable=False)
    adj_offensive_efficiency = Column(Float(10), nullable=False)
    adj_defensive_efficiency = Column(Float(10), nullable=False)
    adj_efficiency_margin = Column(Float(10), nullable=False)
    
    relationship('Games', backref='adjusted_metrics')

class NNModel(Base):
    __tablename__ = 'nn_model'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    scalar_data = Column(LargeBinary, nullable=False)  # The BLOB column for serialized scaler data
    model_data = Column(LargeBinary, nullable=False)
    params = Column(Text, nullable=False)
    train_date = Column(DateTime, nullable=False)

class RidgeMetrics(Base):
    __tablename__ = 'ridge_metrics'
    id = Column(Integer, primary_key=True)
    metric = Column(String, nullable=False)

class RidgeResults(Base):
    __tablename__ = 'ridge_results'
    id = Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey('season.id'), nullable=False)
    metric_id = Column(Integer, ForeignKey('ridge_metrics.id'), nullable=False)
    opponent_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    opponent_value = Column(Float(10), nullable=False)
    conference_id = Column(Integer, ForeignKey('conferences.id'), nullable=False)
    conference_value = Column(Float(10), nullable=False)
    home_value = Column(Float(10), nullable=False)
    away_value = Column(Float(10), nullable=False)

class Predictions(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    team1_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    team2_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    team1_pts = Column(Float(5), nullable=False)
    team2_pts = Column(Float(5), nullable=False)
    winner = Column(Integer, ForeignKey('teams.id'), nullable=False)
    win_pct = Column(Float(5), nullable=False)
    win_margin = Column(Float(5), nullable=False)
    total_pts = Column(Float(5), nullable=False)

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    correct = Column(Integer, nullable=False)
    total = Column(Integer, nullable=False)
    accuracy = Column(Float(5), nullable=False)
    error = Column(Float(5), nullable=False)

if __name__ == '__main__':
    # Create tables
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    session.commit()