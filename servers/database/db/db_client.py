__all__ = ["get_db_service", "recreate_db"]


import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime


__db_service = None

class DBService:
    def __init__(self, db_string : str ):
        
        self.__engine    = create_engine(db_string, pool_size=50, max_overflow=0)
        self.__session   = sessionmaker(bind=self.__engine,autocommit=False, autoflush=False)
        self.db_string   = db_string

    def user(self, user_id : str) -> DBUser:
        return DBUser( user_id, self.__session)
    
    def image(self, image_id : str) -> ImageDB:
        return ImageDB( image_id, self.__session)

    def dataset(self, dataset_id : str) ->DatasetDB:
        return DatasetDB( dataset_id, self.__session)

    def check_user_existence( self, user_id : str ) -> bool:
        return self.user(user_id).check_existence()

    def check_image_existence( self, image_id : str ) -> bool:
            return self.image(image_id).check_existence()
    
    def check_dataset_existence( self, dataset_id : str ) -> bool:
            return self.dataset(dataset_id).check_existence()
    
    def check_user_existence_by_name( self, username : str ) -> bool:
        session = self.__session()
        try:
           user = session.query( 
                    session.query(User).filter_by(username=username).exists() 
           ).scalar()
           return user
        finally:
            session.close()  

    def fetch_user_from_token( self, token : str) -> str:
        session = self.__session()
        try:
           user = session.query(User).filter_by(token=token).one()
           return user.user_id
        finally:
            session.close()

    def fetch_dataset_from_name( self,  name : str) -> str:
        session = self.__session()
        try:
           dataset = session.query(Dataset).filter_by(name=name).one()
           return dataset.dataset_id
        finally:
            session.close() 



#
# get database service
#
def get_db_service( db_string : str=os.environ.get("DB_STRING","")):
    global __db_service
    if not __db_service:
        __db_service = DBService(db_string)
    return __db_service


def recreate_db():

    db_service = get_db_service()
    Base.metadata.drop_all(db_service.engine())
    Base.metadata.create_all(db_service.engine())

    # NOTE: this should be removed in future!
    # set the root user for development
    user_id = random_id()
    token   = "60f3e0f934054b0faf9ff1d0c27e0fdf790b93702ec94c05a49a30d8dd7d7cdc"
    root_user = User( user_id     = user_id,
                      token       = token,
                      first_name  ="admin",
                      last_name   ="",
                      username    ='jodafons',
                      email       ="jodafons@qio.dell.com",
                      company_name="DELL technology")
    root_user.role = UserRole.SUPER_USER
    db_service.save_user(root_user)