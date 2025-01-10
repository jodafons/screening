

class User(Base):

    __tablename__ = "user"
    id             = Column(Integer, primary_key = True)
    userid         = Column(String, unique=True)
    username       = Column(String)
    full_name      = Column(String)
    email          = Column(String)
    hashed_password= Column(String)
    disabled       = Column(Boolean, default=False)
    level          = Column(String)




__all__ = ["DBUser", "User", "UserStatus" , "UserRole"]

import enum

from sqlalchemy import Column, Integer, String
from sqlalchemy import Enum as SQLEnum
from dataclasses import dataclass
from . import Base


class UserStatus(enum.Enum):
    ACTIVE  = "active"
    BLOCKED = "blocked"

class UserRole(enum.Enum):
    USER        = "user"      
    SUPER_USER  = "super_user"  
    ADMIN       = "admin"

@dataclass
class User(Base):

    __tablename__ = "user"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(String(64))
    token         = Column(String(64))
    username      = Column(String(64), unique=True)
    full_name     = Column(String)
    email         = Column(String)
    role          = Column(SQLEnum(UserRole)  , default=UserRole.USER )
    status        = Column(SQLEnum(UserStatus), default=UserStatus.ACTIVE )




class DBUser:
    
    def __init__(self, user_id : str, session):
        self.__session = session
        self.user_id   = user_id

    def check_existence(self):
        session = self.__session()
        try:
           user = session.query( 
                    session.query(User).filter_by(user_id=self.user_id).exists() 
           ).scalar()
           return user
        finally:
            session.close()

    def check_token(self, token : str) -> bool:
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           return token==user.token
        finally:
            session.close()

    def update_token(self, new_token : str):
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           user.token=new_token
           session.commit()
        finally:
            session.close()

    def update_status(self, status : UserStatus ):
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           user.status=status
           session.commit()
        finally:
            session.close()

    def fetch_token(self) -> str:
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           return user.token
        finally:
            session.close()

    def fetch_full_name(self) -> str:
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()

           return user.full_name
        finally:
            session.close()

    def fetch_status(self) -> UserStatus:
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           return user.status
        finally:
            session.close()

    def fetch_role(self) -> UserRole:
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           return user.role
        finally:
            session.close()

    def fetch_username(self) -> UserRole:
        session = self.__session()
        try:
           user = session.query(User).filter_by(user_id=self.user_id).one()
           return user.username
        finally:
            session.close()



    def is_active(self):
        return self.fetch_status()==UserStatus.ACTIVE

    def deactivate(self):
        self.update_status(UserStatus.BLOCKED)

    def activate(self):
        self.update_status(UserStatus.ACTIVE)